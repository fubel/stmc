import statistics
import time
from typing import Any, List, Optional, Tuple

import motmetrics as mm
import torch
from omegaconf import DictConfig
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

from .similarities import batch_bev_distance, batch_cosine_similarity, batched_box_iou
from .solver import multicut, scale_weights
from .supertrack import SuperTrack, TrackState


class Tracker:
    def __init__(
        self,
        solver_opts: Any,
        cfg: DictConfig,
        n_cams: int,
        feature_extractor: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = "cpu",
    ):
        """
        Initialize the Tracker.

        Args:
            solver_opts: Options for the solver.
            cfg: Configuration dictionary.
            n_cams: Number of cameras.
            feature_extractor: Feature extractor module.
            device: Device to run the tracker on.
        """
        self.feature_extractor = feature_extractor
        self.solver_opts = solver_opts
        self.device = device

        self.current_data = None

        self.feature_dim = cfg.tracker.fdim
        self.n_cams = n_cams
        self.cfg = cfg.tracker

        self.tracks: List[SuperTrack] = []

        self.frame = 0
        self.free_id = 1

        self.latency = []

        self.update_interval = 1
        self.stats = {
            "# Killed": 0,
            "Latency": 0,
        }

        self.cumulative_execution_time = 0

    def step(self, sample):
        """
        Perform a single step of tracking.

        Args:
            sample: Input sample containing detections and features.

        Returns:
            tuple: A tuple containing current results and predicted results.
        """
        # move sample to device and remove batch dimension
        t0 = time.time()
        for key in sample.keys():
            if key != "images":
                sample[key] = sample[key].to(self.device).squeeze(0)
        self.frame += 1
        if self.frame % self.update_interval == 0:
            if sample["annotations"].size(0) > 0:
                matched, unmatched = self.update(sample)
                self._handle_unmatched(unmatched)

        t1 = time.time()
        self.cumulative_execution_time += t1 - t0
        self.latency.append(t1 - t0)

        self._sanitize()

        rresults = self.get_result()

        self.predict()

        presults = self.get_result()

        return rresults, presults

    def update(self, sample):
        """
        Update the tracker with new detections and features.

        Args:
            sample: Input sample containing detections and features.

        Returns:
            tuple: A tuple containing matched and unmatched tracks.
        """
        features = self.feature_extractor(sample)
        superboxes = self._new_superboxes_from_data(sample, features)
        superboxes = [s for s in superboxes if s.confidence >= self.cfg.confidence_thresh]

        relevant_tracks = self.tracks + superboxes
        _track_indices = torch.arange(len(self.tracks)).to(self.device)
        _superbox_indices = torch.arange(len(self.tracks), len(relevant_tracks)).to(self.device)

        low_conf_indices = None

        if self.cfg.low_confidence_thresh is not None:
            c1 = self.cfg.low_confidence_thresh
            c2 = self.cfg.confidence_thresh
            low_conf_superboxes = [s for s in superboxes if c1 <= s.confidence < c2]

            if len(low_conf_superboxes) > 0:
                n_relevant = len(relevant_tracks)
                relevant_tracks += low_conf_superboxes
                low_conf_indices = torch.arange(n_relevant, n_relevant + len(low_conf_superboxes))

        if len(relevant_tracks) == 0:
            return [], []

        features = torch.stack([track.p_features for track in relevant_tracks])  # (n_tracks, n_cams, feature_dim)
        positions = torch.stack([track.p_positions for track in relevant_tracks])  # (n_tracks, n_cams, 2)
        boxes = torch.stack([track.tlbr for track in relevant_tracks])  # (n_tracks, n_cams, 4)

        # compute (n_tracks) x (n_tracks) similarity matrix
        similarities = self._compute_similarities(features, positions, boxes)

        # compute weighted graph
        rescale_thresh = self.cfg.matching.rescale_threshold
        dist_thresh = self.cfg.matching.distance_threshold
        iou_bias = self.cfg.prematching.iou_bias if self.cfg.prematching.enabled else 0
        edge_index, edge_weights = self._build_weighted_graph(
            relevant_tracks,
            similarities,
            rescale_thresh,
            dist_thresh,
            iou_bias,
            reid_decay=self.cfg.matching.reid_decay,
        )
        labels = multicut(edge_index, edge_weights, self.solver_opts)

        matched_tracks, unmatched_tracks = self._match(relevant_tracks, labels, low_conf_indices=low_conf_indices)

        self.tracks = matched_tracks + unmatched_tracks
        return matched_tracks, unmatched_tracks

    def _handle_unmatched(self, unmatched_tracks):
        """
        Handle unmatched tracks by updating their inactive status.

        Args:
            unmatched_tracks: List of unmatched tracks.
        """
        for track in unmatched_tracks:
            for cam in range(self.n_cams):
                if track.track_where[cam]:
                    track.inactive_since[cam] += 1

    def predict(self):
        """
        Project existing tracks into the future.
        """
        for track in self.tracks:
            track.predict()

    def _new_superboxes_from_data(self, sample, sample_features):
        """
        Create new superboxes from detections and features.

        Args:
            sample: Input sample containing detection information.
            sample_features: Extracted features from the sample.

        Returns:
            list: List of new SuperTrack objects.
        """
        n_rows = sample_features.shape[0]

        features = torch.full((n_rows, self.n_cams, self.feature_dim), float("nan"), device=self.device)
        boxes = torch.full((n_rows, self.n_cams, 4), float("nan"), device=self.device)
        positions_2d = torch.full((n_rows, self.n_cams, 2), float("nan"), device=self.device)
        positions_3d = torch.full((n_rows, self.n_cams, 2), float("nan"), device=self.device)

        cam_ids = sample["annotations"][:, 0].int()
        features[torch.arange(n_rows), cam_ids] = sample_features
        boxes[torch.arange(n_rows), cam_ids] = sample["annotations"][:, 3:7]
        positions_2d[torch.arange(n_rows), cam_ids] = sample["positions_2d"]
        positions_3d[torch.arange(n_rows), cam_ids] = sample["positions_3d"]
        confidences = sample["annotations"][:, 7]

        superboxes = [
            SuperTrack(
                frame=self.frame,
                features=features[row],
                boxes=boxes[row],
                positions_2d=positions_2d[row],
                positions_3d=positions_3d[row],
                confidence=confidences[row],
            )
            for row in range(n_rows)
        ]

        return superboxes

    def _merge_tracks(self, tracks):
        """
        Merge multiple tracks into a single track.

        Args:
            tracks: List of tracks to merge.

        Returns:
            SuperTrack: Merged track.
        """
        _frames = sorted({track.frame for track in tracks})

        newest_frame = _frames[-1]
        if len(_frames) > 1:
            penult_frame = _frames[-2]

        assert tracks[-1].frame == newest_frame

        newest_evidence = [track for track in tracks if track.frame == newest_frame]

        features = (torch.ones(self.n_cams, self.feature_dim) * (torch.nan)).to(self.device)
        boxes = (torch.ones(self.n_cams, 4) * (torch.nan)).to(self.device)
        positions_2d = (torch.ones(self.n_cams, 2) * (torch.nan)).to(self.device)
        positions_3d = (torch.ones(self.n_cams, 2) * (torch.nan)).to(self.device)
        track_where = torch.zeros(self.n_cams, dtype=torch.bool).to(self.device)

        for cam_id in range(self.n_cams):
            for track in newest_evidence:
                if not torch.isnan(track.features[cam_id]).any():
                    features[cam_id] = track.features[cam_id]
                    boxes[cam_id] = track.boxes[cam_id]
                    positions_2d[cam_id] = track.positions_2d[cam_id]
                    positions_3d[cam_id] = track.positions_3d[cam_id]
                    track_where[cam_id] = True
                    break

        merged_track = SuperTrack(
            frame=newest_frame,
            features=features,
            boxes=boxes,
            positions_2d=positions_2d,
            positions_3d=positions_3d,
        )

        if len(_frames) > 1:
            penult_track = [track for track in tracks if track.frame == penult_frame][0]
            penult_track.update(merged_track)
            merged_track = penult_track

        return merged_track

    def _match(self, tracks, labels, low_conf_indices=None):
        """
        Match superboxes with superboxes, and merged superboxes with existing supertracks.

        Args:
            tracks: List of tracks to match.
            labels: Labels for each track.
            low_conf_indices: Indices of low confidence detections.

        Returns:
            tuple: A tuple containing new tracks and unmatched tracks.
        """
        new_tracks = []
        unmatched_tracks = []

        for label in torch.unique(labels):
            track_indices = torch.where(labels == label)[0].tolist()
            if len(track_indices) == 1:
                track = tracks[track_indices[0]]
                if low_conf_indices is not None and track_indices[0] in low_conf_indices:
                    continue
                if track.state == TrackState.CREATED:
                    new_tracks.append(track)
                else:
                    unmatched_tracks.append(track)
            else:
                if low_conf_indices is None:
                    relevant_tracks = sorted([tracks[i] for i in track_indices], key=lambda x: x.frame)
                else:
                    relevant_tracks = sorted(
                        [tracks[i] for i in track_indices if i not in low_conf_indices], key=lambda x: x.frame
                    )
                merged_track = self._merge_tracks(relevant_tracks)
                if low_conf_indices is not None and not merged_track.is_complete():
                    relevant_low_conf_tracks = [tracks[i] for i in track_indices if i in low_conf_indices]
                    merged_track = self._merge_tracks([merged_track] + relevant_low_conf_tracks)
                new_tracks.append(merged_track)

        return new_tracks, unmatched_tracks

    @staticmethod
    def _compute_similarities(features, positions, boxes):
        """
        Compute similarity matrices for features, positions, and boxes.

        Args:
            features: Tensor of track features.
            positions: Tensor of track positions.
            boxes: Tensor of track bounding boxes.

        Returns:
            tuple: A tuple containing similarity matrices for features, positions, and IoU.
        """
        # permute to (n_cams, n_tracks, feature_dim), (n_cams, n_tracks, 2), (n_cams, n_tracks, 4)
        features = features.permute(1, 0, 2)
        positions = positions.permute(1, 0, 2)
        boxes = boxes.permute(1, 0, 2)

        # compute pairwise similarities (n_cams, n_tracks, n_tracks)
        feature_sim = batch_cosine_similarity(features, features)
        position_dist = batch_bev_distance(positions)
        iou_sim = batched_box_iou(boxes)

        # average-pool similarities to (n_tracks, n_tracks)
        feature_sim = torch.nanmean(feature_sim, dim=0)
        position_dist = torch.nanmean(position_dist, dim=0)
        iou_sim = torch.nanmean(iou_sim, dim=0)

        return feature_sim, position_dist, iou_sim

    def _build_weighted_graph(
        self,
        tracks: List[SuperTrack],
        similarities: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        rescale_thresh: float,
        dist_thresh: float,
        iou_bias: float,
        reid_decay: float = 1,
        penalty: float = -100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build a weighted graph from tracks and similarity matrices.

        Args:
            tracks: List of tracks.
            similarities: Tuple of similarity matrices.
            rescale_thresh: Threshold for rescaling weights.
            dist_thresh: Distance threshold for feasibility.
            iou_bias: Bias to add for IoU-based matching.
            reid_decay: Decay factor for ReID scores.
            penalty: Penalty for infeasible edges.

        Returns:
            tuple: A tuple containing edge indices and edge weights of the graph.
        """
        adj = self._initialize_adjacency_matrix(similarities, tracks, reid_decay, rescale_thresh, dist_thresh)

        if self.cfg.prematching.enabled:
            adj = self._apply_prematching(adj, tracks, iou_bias)

        adj = self._finalize_adjacency_matrix(adj, penalty, tracks)

        return self._get_edge_index_and_weights(adj)

    def _initialize_adjacency_matrix(
        self,
        similarities: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tracks: List[SuperTrack],
        reid_decay: float,
        rescale_thresh: float,
        dist_thresh: float,
    ) -> torch.Tensor:
        """
        Initialize the adjacency matrix for the graph.

        Args:
            similarities: Tuple of similarity matrices.
            tracks: List of tracks.
            reid_decay: Decay factor for ReID scores.
            rescale_thresh: Threshold for rescaling weights.
            dist_thresh: Distance threshold for feasibility.

        Returns:
            torch.Tensor: Initialized adjacency matrix.
        """
        appearance_sim, position_dist, _ = similarities
        device = appearance_sim.device

        frame_support_pairs = [(track.frame, track.track_where) for track in tracks]
        frames, supports = zip(*frame_support_pairs)

        times = torch.tensor(frames, dtype=torch.int, device=device)
        lost = torch.tensor([track.state == TrackState.LOST for track in tracks], device=device)
        lost_since = torch.tensor([track.lost_since for track in tracks], device=device)

        appearance_sim = appearance_sim * reid_decay**lost_since
        appearance_sim = scale_weights(appearance_sim, rescale_thresh)

        combined_sim = self.cfg.matching.rescale_weight * appearance_sim + self.cfg.matching.distance_weight * (
            1 - position_dist / dist_thresh
        )

        adj = torch.zeros_like(appearance_sim)
        lmask = lost[:, None] | lost[None, :]
        same_time = times[:, None] == times[None, :]
        feasible = (position_dist < dist_thresh) | lmask

        adj[same_time & feasible] = torch.clip(combined_sim[same_time & feasible], min=0, max=1)
        adj[~same_time] = combined_sim[~same_time]
        adj[lmask] = combined_sim[lmask]

        return adj

    def _apply_prematching(self, adj: torch.Tensor, tracks: List[SuperTrack], iou_bias: float) -> torch.Tensor:
        """
        Apply prematching to the adjacency matrix.

        Args:
            adj: Adjacency matrix.
            tracks: List of tracks.
            iou_bias: Bias to add for IoU-based matching.

        Returns:
            torch.Tensor: Updated adjacency matrix after prematching.
        """
        cur_frame = max(track.frame for track in tracks)
        pen_frame = cur_frame - 1
        cur_track_idx_by_cam = [[] for _ in range(self.n_cams)]
        pen_track_idx_by_cam = [[] for _ in range(self.n_cams)]

        for i, track in enumerate(tracks):
            if track.frame == cur_frame:
                for cam in range(self.n_cams):
                    if not torch.isnan(track.boxes[cam]).any():
                        cur_track_idx_by_cam[cam].append(i)
            elif track.frame == pen_frame:
                for cam in range(self.n_cams):
                    if not torch.isnan(track.boxes[cam]).any():
                        pen_track_idx_by_cam[cam].append(i)

        for cam in range(self.n_cams):
            cur_boxes_cam = [tracks[i].tlbr[cam] for i in cur_track_idx_by_cam[cam]]
            pen_boxes_cam = [tracks[i].tlbr[cam] for i in pen_track_idx_by_cam[cam]]

            if not cur_boxes_cam or not pen_boxes_cam:
                continue

            iou_dist = 1 - box_iou(torch.stack(cur_boxes_cam), torch.stack(pen_boxes_cam))
            row_ind, col_ind = linear_sum_assignment(iou_dist.cpu().numpy())

            for r, c in zip(row_ind, col_ind):
                if iou_dist[r, c] > self.cfg.prematching.iou_threshold:
                    continue
                cur_idx = cur_track_idx_by_cam[cam][r]
                if self.cfg.prematching.prune_remaining:
                    adj[cur_idx] = 0
                    adj[:, cur_idx] = 0
                adj[cur_idx, pen_track_idx_by_cam[cam][c]] += iou_bias
                adj[pen_track_idx_by_cam[cam][c], cur_idx] += iou_bias

        return adj

    def _finalize_adjacency_matrix(self, adj: torch.Tensor, penalty: float, tracks: List[SuperTrack]) -> torch.Tensor:
        """
        Finalize the adjacency matrix by applying penalties.

        Args:
            adj: Adjacency matrix.
            penalty: Penalty value for infeasible edges.
            tracks: List of tracks.

        Returns:
            torch.Tensor: Finalized adjacency matrix.
        """
        frame_support_pairs = [(track.frame, track.track_where) for track in tracks]
        frames, supports = zip(*frame_support_pairs)

        times = torch.tensor(frames, dtype=torch.int, device=adj.device)
        supps = torch.stack(supports).to(adj.device)

        same_time = times[:, None] == times[None, :]
        same_supp = (supps[:, None] & supps[None, :]).any(dim=2)

        adj[same_time & same_supp] = penalty
        adj = adj * torch.triu(torch.ones_like(adj), diagonal=1)

        return adj

    def _get_edge_index_and_weights(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract edge indices and weights from the adjacency matrix.

        Args:
            adj: Adjacency matrix.

        Returns:
            tuple: A tuple containing edge indices and edge weights.
        """
        edge_index = torch.nonzero(adj).t().long()
        edge_weights = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_weights

    def _sanitize(self):
        """
        Sanitize the tracker by updating track states and removing killed tracks.
        """
        keep = []
        for k, track in enumerate(self.tracks):
            if track.state is TrackState.CREATED:
                track.activate()
            if track.label is None:
                track.set_label(self.free_id)
                self.free_id += 1
            if torch.all(~track.track_where):
                if torch.all(track.inactive_since[track.inactive_since > 0] > self.cfg.patience):
                    track.deactivate()
            if track.state is TrackState.LOST:
                if track.lost_since > self.cfg.memory:
                    track.kill()
                else:
                    track.lost_since += 1
            if track.state is not TrackState.KILLED:
                keep.append(track)
            for cam in range(self.n_cams):
                if track.inactive_since[cam] > self.cfg.patience:
                    track.reset([cam])
        killed = len(self.tracks) - len(keep)
        self.tracks = keep
        self.stats["# Tracks"] = len(self.tracks)
        self.stats["# Lost"] = len([track for track in self.tracks if track.state == TrackState.LOST])
        self.stats["# Killed"] += killed

        latency = statistics.mean(self.latency) if len(self.latency) > 0 else 0
        self.stats["FPS"] = int(1 / latency) if latency > 0 else 0

    def _get_active_tracks(self):
        """
        Get a list of active tracks.

        Returns:
            list: List of active tracks.
        """
        return [track for track in self.tracks if track.state != TrackState.KILLED]

    def get_result(self, normalization=None, scale=1.0):
        """
        Get the current online state of the tracker.

        Args:
            normalization: Optional normalization parameters.
            scale: Scale factor for the results.

        Returns:
            torch.Tensor: Tensor containing the current tracker state.
        """
        to_stack = [track.to_tensor() for track in self.tracks if track.state == TrackState.ACTIVE]
        if len(to_stack) > 0:
            result = torch.cat(to_stack)
        else:
            result = torch.empty(0)
        if result.size(0) > 0:
            if normalization is not None:
                min_x, min_y, max_x, max_y = normalization
                result[:, 7:9] = result[:, 7:9] * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor(
                    [min_x, min_y]
                )
            result[:, 7:9] *= scale
        return result

    def _get_index_by_id(self, tid):
        """
        Get the index of a track by its ID.

        Args:
            tid: Track ID to search for.

        Returns:
            int: Index of the track with the given ID, or None if not found.
        """
        for i, track in enumerate(self.tracks):
            if track.label == tid:
                return i
        return None


def create_tracker(cfg, solver_cfg, feature_extractor, n_cams, device, writer=None):
    """
    Create a new Tracker instance.

    Args:
        cfg: Configuration dictionary.
        solver_cfg: Solver configuration.
        feature_extractor: Feature extractor module.
        n_cams: Number of cameras.
        device: Device to run the tracker on.
        writer: Optional writer for logging.

    Returns:
        Tracker: A new Tracker instance.
    """
    return Tracker(
        solver_opts=solver_cfg,
        cfg=cfg,
        feature_extractor=feature_extractor,
        n_cams=n_cams,
        device=device,
    )
