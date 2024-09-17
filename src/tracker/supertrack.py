from enum import IntEnum

import torch

from ..utils.utils import tlwh_to_tlbr


class TrackState(IntEnum):
    CREATED = 0  # Track is created but not confirmed yet
    ACTIVE = 1  # Track is confirmed and active
    LOST = 3  # Track is lost and not tracked, but kept in memory
    KILLED = 4  # Track is killed (e.g. due to merging with another track)


class SuperTrack:
    def __init__(
        self,
        frame,
        features,
        boxes,
        positions_2d,
        positions_3d,
        confidence=None,
    ):
        self.frame = frame
        self.last_update = frame

        self.n_cams = features.size(0)
        self.features = features
        self.boxes = boxes
        self.positions_2d = positions_2d
        self.positions_3d = positions_3d

        self.label = None
        self.__state = TrackState.CREATED  # private state variable

        # inactivity counter: how many frames since last update at each camera
        self.inactive_since = torch.zeros(self.n_cams, device=features.device)

        self.lost_since = 0

        # where to continue tracking: if False, track is not continued in this camera
        self.track_where = torch.ones(self.n_cams, device=features.device).bool()
        self.track_where[torch.isnan(features).any(dim=1)] = False

        # cams the track hasn't been seen in yet
        self.queries = torch.ones(self.n_cams, device=features.device).bool()

        # count updates for each camera
        self.ticks = torch.ones(self.n_cams, device=features.device)

        self.confidence = confidence

        self.velocities_2d = torch.zeros((self.n_cams, 4), device=features.device)
        self.velocities_3d = torch.zeros((self.n_cams, 2), device=features.device)

    @classmethod
    def empty(cls, n_cams, fdim, device):
        return cls(
            frame=None,
            features=torch.full((n_cams, fdim), float("nan"), device=device),
            boxes=torch.full((n_cams, 4), float("nan"), device=device),
            positions_2d=torch.full((n_cams, 2), float("nan"), device=device),
            positions_3d=torch.full((n_cams, 3), float("nan"), device=device),
        )

    def activate(self):
        self.__state = TrackState.ACTIVE

    def deactivate(self):
        self.__state = TrackState.LOST

    def kill(self):
        self.__state = TrackState.KILLED

    def reset(self, cams=None):
        if cams is None:
            cams = range(self.n_cams)
        for cam in cams:
            self.track_where[cam] = False
            # self.inactive_since[cam] = 0

    def set_label(self, label):
        if self.label is not None:
            raise ValueError(f"Track {self} is already labeled.")
        self.label = label

    @property
    def keys(self):
        return ~self.queries

    @property
    def state(self):
        return self.__state

    @property
    def tlbr(self):
        return tlwh_to_tlbr(self.boxes)

    def is_complete(self):
        return ~torch.isnan(self.features).any()

    @property
    def p_features(self):
        return self.phantomize(self.features)

    @property
    def p_positions(self):
        return self.phantomize(self.positions_3d)

    @property
    def mean_positions_3d(self):
        return torch.nanmean(self.positions_3d, dim=0)

    @staticmethod
    def phantomize(tensor):
        """
        Given a (B, n_cams, f_dim) tensor, replace nans with the average of
        the non-nan values along the cam axis.
        """
        return torch.where(torch.isnan(tensor), torch.nanmean(tensor, dim=0, keepdim=True), tensor)

    def update(self, other):
        n_cams = self.features.size(0)
        if self.frame == other.frame:
            for cam in range(n_cams):
                if torch.isnan(self.features[cam]).any():
                    if torch.isnan(other.features[cam]).any():
                        continue
                    self.features[cam] = other.features[cam]
                    self.boxes[cam] = other.boxes[cam]
                    self.positions_2d[cam] = other.positions_2d[cam]
                    self.positions_3d[cam] = other.positions_3d[cam]
                    self.inactive_since[cam] = 0
                    self.track_where[cam] = True
                    self.queries[cam] = False
                    self.ticks[cam] = other.ticks[cam]
                else:
                    if not torch.isnan(other.features[cam]).any():
                        raise ValueError(f"Found violation of constraints for track update with {self}.")
        elif self.frame < other.frame:
            for cam in range(n_cams):
                if not torch.isnan(other.features[cam]).any():
                    if not torch.isnan(self.features[cam]).any():
                        if self.velocities_2d[cam].sum() == 0:
                            w = 1.0
                        else:
                            w = 0.8
                        self.velocities_2d[cam] = (
                            w * (other.boxes[cam] - self.boxes[cam]) / (other.frame - self.frame)
                            + (1 - w) * self.velocities_2d[cam]
                        )
                        self.velocities_3d[cam] = (
                            w * (other.positions_3d[cam] - self.positions_3d[cam]) / (other.frame - self.frame)
                            + (1 - w) * self.velocities_3d[cam]
                        )
                        self.features[cam] = 0.9 * self.features[cam] + 0.1 * other.features[cam]
                        self.boxes[cam] = other.boxes[cam]
                        self.positions_2d[cam] = other.positions_2d[cam]
                        self.positions_3d[cam] = other.positions_3d[cam]
                        self.inactive_since[cam] = 0
                        self.track_where[cam] = True
                        self.queries[cam] = False
                        self.ticks[cam] += 1
                    else:
                        self.features[cam] = other.features[cam]
                        self.boxes[cam] = other.boxes[cam]
                        self.positions_2d[cam] = other.positions_2d[cam]
                        self.positions_3d[cam] = other.positions_3d[cam]
                        self.inactive_since[cam] = 0
                        self.track_where[cam] = True
                        self.queries[cam] = False
                        self.ticks[cam] = other.ticks[cam]
                else:
                    if self.track_where[cam]:
                        self.inactive_since[cam] += 1
        else:
            raise ValueError(
                f"Frame of other must be greater or equal to frame of self, but got {self.frame} and {other.frame}."
            )
        self.last_update = other.frame
        self.frame = other.frame

        if self.state == TrackState.LOST:
            self.activate()

    def predict(self):
        for cam in range(self.n_cams):
            if ~self.track_where[cam]:
                continue
            prd_box = self.boxes[cam] + self.velocities_2d[cam]
            prd_pos = self.positions_3d[cam] + self.velocities_3d[cam]
            if prd_box[2] <= 0 or prd_box[3] <= 0:
                prd_box = self.boxes[cam]
                prd_pos = self.positions_3d[cam]
            self.boxes[cam] = prd_box
            self.positions_3d[cam] = prd_pos

    def merge(self, other):
        if other.state == TrackState.KILLED or self.state == TrackState.KILLED:
            raise ValueError("Cannot merge killed tracks.")
        if other.frame < self.frame:
            raise ValueError(
                f"Other track must not be older than self, but "
                f"self is at frame {self.frame} and other at frame {other.frame}."
            )
        self.update(
            other.frame,
            other.features,
            other.boxes,
            other.positions_2d,
            other.positions_3d,
        )
        # other was merged into self, so it is killed
        other.kill()

    def split(self, where: torch.Tensor):
        # keep the cams where "where" is True
        other_features = self.features.clone()
        other_boxes = self.boxes.clone()
        other_positions_2d = self.positions_2d.clone()
        other_positions_3d = self.positions_3d.clone()
        for w in where:
            if not w:
                self.features[w] = torch.nan
                self.boxes[w] = torch.nan
                self.positions_2d[w] = torch.nan
                self.positions_3d[w] = torch.nan
            else:
                other_features[w] = torch.nan
                other_boxes[w] = torch.nan
                other_positions_2d[w] = torch.nan
                other_positions_3d[w] = torch.nan
        return SuperTrack(
            frame=self.frame,
            features=other_features,
            boxes=other_boxes,
            positions_2d=other_positions_2d,
            positions_3d=other_positions_3d,
        )

    def __repr__(self):
        return f"Track {self.label}"

    def to_tensor(self):
        output = []
        if self.state == TrackState.LOST:
            return torch.Tensor(output)
        for i, box in enumerate(self.boxes):
            if ~self.track_where[i]:
                continue
            row = [i, self.label, self.frame, *box, *self.mean_positions_3d]
            output.append(row)
        return torch.Tensor(output)
