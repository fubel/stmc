import glob
import os
import pathlib
import warnings
from enum import IntEnum
from typing import List, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.ops import nms

from ..tracker.geometry import Projector
from ..utils.utils import compute_centers, resize_transform, tlwh_to_tlbr


class Annotation(IntEnum):
    CAM_ID = 0
    OBJ_ID = 1
    FRAME_ID = 2
    XMIN = 3
    YMIN = 4
    WIDTH = 5
    HEIGHT = 6
    CONF = 7
    XWORLD = 8
    YWORLD = 9


class NMSTransform:
    def __init__(self, iou_threshold: float):
        """Initialize the NMSTransform which applied non-maximum suppression to the
        input annotations based on the specified IoU threshold.

        Args:
            iou_threshold (float): The Intersection over Union (IoU) threshold for NMS.
                Bounding boxes with IoU greater than this threshold will be suppressed.
        """
        self.iou_threshold = iou_threshold

    def __call__(self, annotations: torch.Tensor) -> torch.Tensor:
        boxes = tlwh_to_tlbr(annotations[:, Annotation.XMIN : Annotation.HEIGHT + 1])
        scores = annotations[:, Annotation.CONF]
        keep = nms(boxes, scores, self.iou_threshold)
        return keep


class ROIFilter:
    def __init__(self, roi_path: str):
        """Initialize the ROIFilter.

        Args:
            roi_path (str): Path to the ROI image file.

        The ROI (Region of Interest) image is loaded as a binary mask,
        where 1 indicates areas of interest and 0 indicates areas to be filtered out.
        """
        self.roi = read_image(roi_path, ImageReadMode.GRAY).squeeze(0).bool()
        self.size = self.roi.size()

    def __call__(self, annotations: torch.Tensor) -> torch.Tensor:
        centers = compute_centers(annotations[:, Annotation.XMIN - 1 : Annotation.HEIGHT]).int()
        centers[:, 0] = torch.clamp(centers[:, 0], 0, self.size[1] - 1)
        centers[:, 1] = torch.clamp(centers[:, 1], 0, self.size[0] - 1)
        keep = self.roi[centers[:, 1], centers[:, 0]] == 1
        return keep


class MultiCamDataset:
    def __init__(
        self,
        annotation_paths: List[str],
        image_paths: List[str],
        calibration_paths: List[str],
        camera_names: List[int],
        ground_truth_paths: Optional[List[str]] = None,
        precomputed: bool = False,
        nms_threshold: Optional[float] = 0.9,
        time_offsets: Optional[List[int]] = None,
        roi_paths: Optional[List[str]] = None,
        normalize_bev: bool = False,
        bottom: bool = True,
        box_projection_centers=None,
    ):
        """Initialize the MultiCamDataset for data loading.

        Args:
            annotation_paths (List[str]): Paths to annotation files for each camera.
            image_paths (List[str]): Paths to image directories for each camera.
            calibration_paths (List[str]): Paths to calibration files for each camera.
            camera_names (List[int]): Names or IDs of the cameras.
            ground_truth_paths (Optional[List[str]], optional): Paths to ground truth files. Defaults to None.
            precomputed (bool, optional): Whether to use precomputed features. Defaults to False.
            nms_threshold (Optional[float], optional): Non-maximum suppression threshold. Defaults to 0.9.
            time_offsets (Optional[List[int]], optional): Time offsets for each camera. Defaults to None.
            roi_paths (Optional[List[str]], optional): Paths to region of interest mask images. Defaults to None.
            normalize_bev (bool, optional): Whether to normalize bird's-eye view coordinates. Defaults to False.
            bottom (bool, optional): Whether to use bottom of bounding box for projection. Defaults to True.
            box_projection_centers (Optional[Tuple[float, float]], optional): Projection centers for bounding boxes. Defaults to None.
        """
        if time_offsets is None:
            self.time_offsets = [0] * len(image_paths)
        else:
            self.time_offsets = time_offsets

        self.annotation_paths = annotation_paths
        self.image_paths = image_paths
        self.calibration_paths = calibration_paths
        self.camera_names = camera_names
        self.precomputed = precomputed
        self.nms_transform = NMSTransform(nms_threshold) if nms_threshold is not None else None
        self.box_projection_centers = box_projection_centers
        self.bottom = bottom

        self.normalize_bev = normalize_bev

        if roi_paths is not None:
            self.roi_filters = [ROIFilter(roi_path) for roi_path in roi_paths]
        else:
            self.roi_filters = None

        self._load_calibrations()
        self._load_annotations()

        if ground_truth_paths is not None:
            self._load_ground_truth(ground_truth_paths)
        else:
            self._ground_truths = None
            self.gts = None

        self.length = max([len(list(pathlib.Path(image_path).glob("*.jpg"))) for image_path in self.image_paths])

        if self.length == 0:
            warnings.warn("No images found. Visualization tools will not be available.")

        self.length = 2110

        self._filtered_by_nms = 0
        self._filtered_by_size = 0
        self._filtered_by_roi = 0

    def _load_ground_truth(self, ground_truth_paths):
        self._ground_truths = [
            torch.from_numpy(np.loadtxt(ground_truth_path, delimiter=",", dtype=np.float32))
            for ground_truth_path in ground_truth_paths
        ]

        for gt in self._ground_truths:
            if gt.shape[1] == 9:
                # append another column of ones
                gt = torch.cat((gt, torch.ones(gt.shape[0], 1)), dim=1)

        _cat_gts = [g.clone() for g in self._ground_truths]
        for i, gt in enumerate(_cat_gts):
            col = torch.ones((gt.shape[0], 1)) * i
            _cat_gts[i] = torch.cat((col, gt), dim=1)
            _cat_gts[i][:, 1] += self.time_offsets[i]

        self.gts = torch.cat(_cat_gts, dim=0)
        self.gts[:, [1, 2]] = self.gts[:, [2, 1]]

    def _load_calibrations(self):
        self._projectors = [Projector(calibration_path) for calibration_path in self.calibration_paths]

    def _load_annotations(self):
        anns = [
            torch.from_numpy(np.loadtxt(annotation_path, delimiter=",", dtype=np.float32))
            for annotation_path in self.annotation_paths
        ]

        # todo: add to preprocess config
        for i, ann in enumerate(anns):
            keep = (ann[:, Annotation.WIDTH - 1] * ann[:, Annotation.HEIGHT - 1]) >= 1200
            anns[i] = ann[keep]

        # filter roi images
        if self.roi_filters is not None:
            keep = self.roi_filters[i](anns[i])
            anns[i] = anns[i][keep]
            logger.info(f"🔥 Filtered {keep.size(0) - keep.sum().item()} annotations by ROI.")

        for i, ann in enumerate(anns):
            col = torch.ones((ann.shape[0], 1)) * i
            anns[i] = torch.cat((col, ann), dim=1)
            anns[i][:, 1] += self.time_offsets[i]

        positions_2d = []
        for i, ann in enumerate(anns):
            pos2d = compute_centers(
                ann[:, Annotation.XMIN : Annotation.HEIGHT + 1], self.bottom, self.box_projection_centers
            )
            positions_2d.append(pos2d)

        positions_3d = []
        for i, pos2d in enumerate(positions_2d):
            pos3d = self._projectors[i].image_to_world(pos2d)
            positions_3d.append(pos3d)

        anns = torch.cat(anns, dim=0)
        positions_2d = torch.cat(positions_2d, dim=0)
        positions_3d = torch.cat(positions_3d, dim=0)

        if anns.shape[1] == 9:
            # loaded from ground truth, append column of 1s as 7th column
            anns = torch.cat(
                (
                    anns[:, :6],
                    torch.ones(anns.shape[0], 1),
                    anns[:, 6:],
                ),
                dim=1,
            )
        # swap columns frame and obj_id
        anns[:, [1, 2]] = anns[:, [2, 1]]

        self._annotations = anns
        self._positions_2d = positions_2d
        self._positions_3d = positions_3d

        if self.normalize_bev:
            self.apply_bev_norm()
        else:
            self._norm_factors = None

        self._annotations.to("cuda")
        self._positions_2d.to("cuda")
        self._positions_3d.to("cuda")

    def get_bev_ticks(self):
        return [
            float(torch.min(self._positions_3d[:, 0])),
            float(torch.max(self._positions_3d[:, 0])),
            float(torch.min(self._positions_3d[:, 1])),
            float(torch.max(self._positions_3d[:, 1])),
        ]

    def get_crops(self, frame_annotations, frame_images):
        crops = []
        for ann in frame_annotations:
            cam_id = int(ann[Annotation.CAM_ID])
            x, y, w, h = ann[Annotation.XMIN : Annotation.HEIGHT + 1].int()
            # clamp to image dimensions
            x = torch.clamp(x, 0, frame_images[cam_id].size(1) - 1)
            y = torch.clamp(y, 0, frame_images[cam_id].size(2) - 1)
            w = torch.clamp(w, 0, frame_images[cam_id].size(1) - x)
            h = torch.clamp(h, 0, frame_images[cam_id].size(2) - y)
            crops.append(resize_transform(frame_images[cam_id][:, y : y + h, x : x + w]))
        if len(crops) == 0:
            return torch.empty(0)
        return torch.stack(crops)

    def apply_bev_norm(self):
        # normalize BEV positions to [0, 1]
        logger.info("📏 Normalizing BEV positions to [0, 1].")
        min_x, min_y = torch.min(self._positions_3d, dim=0)[0]
        max_x, max_y = torch.max(self._positions_3d, dim=0)[0]
        self._norm_factors = torch.tensor([min_x, min_y, max_x, max_y])
        self._positions_3d = (self._positions_3d - torch.tensor([min_x, min_y])) / torch.tensor(
            [max_x - min_x, max_y - min_y]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        frame = idx + 1

        annotations = self._annotations[self._annotations[:, Annotation.FRAME_ID] == frame]
        positions_2d = self._positions_2d[self._annotations[:, Annotation.FRAME_ID] == frame]
        positions_3d = self._positions_3d[self._annotations[:, Annotation.FRAME_ID] == frame]

        if self.gts is not None:
            ground_truth = self.gts[self.gts[:, Annotation.FRAME_ID] == frame]
        else:
            ground_truth = torch.empty(0)

        if self.nms_transform is not None:
            keep = self.nms_transform(annotations)
        else:
            keep = torch.arange(annotations.size(0))

        annotations = annotations[keep]
        positions_2d = positions_2d[keep]
        positions_3d = positions_3d[keep]

        frame_images = []
        for img_path, offset in zip(self.image_paths, self.time_offsets):
            try:
                frame_images.append(read_image(str(pathlib.Path(img_path) / f"{(frame - offset):06d}.jpg")))
            except Exception:
                frame_images.append(torch.zeros(3, 1080, 1920).to(torch.uint8))

        if not self.precomputed:
            frame_crops = self.get_crops(annotations, frame_images)
        else:
            frame_crops = torch.empty(0)

        return {
            "annotations": annotations,
            "positions_2d": positions_2d,
            "positions_3d": positions_3d,
            "images": frame_images,
            "crops": frame_crops,
            "ground_truth": ground_truth,
        }


def create_dataloader(cfg):
    scene_path = os.path.join(cfg.dataset_path, cfg.dataset.scene_path)
    cameras = [
        os.path.basename(f)
        for f in sorted(glob.glob(os.path.join(scene_path, cfg.dataset.camera_pattern)))
        if os.path.isdir(f)
    ]

    img_paths = [
        os.path.join(cfg.dataset_path, cfg.dataset.scene_path, camera, cfg.dataset.img_path) for camera in cameras
    ]
    calibration_paths = [
        os.path.join(
            cfg.dataset_path,
            cfg.dataset.scene_path,
            camera,
            cfg.dataset.calibration_path,
        )
        for camera in cameras
    ]
    annotation_paths = []
    for camera in cameras:
        if cfg.resources.reid is not None:
            scene_path = "-".join(pathlib.Path(cfg.dataset.scene_path).parts)
            if scene_path[-1] == "-":
                scene_path = scene_path[:-1]
            resource_name = (
                f"{cfg.dataset.name}_{scene_path}-{camera}_{cfg.resources.detector}_{cfg.resources.reid}.txt"
            )
        else:
            resource_name = f"{cfg.dataset.name}-{camera}_{cfg.resources.detector}.txt"
        annotation_paths.append(os.path.join(cfg.resources.path, resource_name))

    if cfg.preprocess.nms_thresh is not None:
        nms_threshold = cfg.preprocess.nms_thresh
    else:
        nms_threshold = None

    if cfg.preprocess.roi_filter is not None and "roi_path" in cfg.dataset:
        roi_paths = [os.path.join(cfg.dataset.roi_path, camera, "roi.jpg") for camera in cameras]
    else:
        roi_paths = None

    ground_truth_paths = None

    time_offsets = None
    if "offsets" in cfg.dataset:
        if cfg.dataset.offsets is not None:
            time_offsets = cfg.dataset.offsets

    box_projection_centers = [
        cfg.preprocess.box_projection_centers.alpha_w,
        cfg.preprocess.box_projection_centers.alpha_h,
    ]

    if box_projection_centers[0] is None:
        box_projection_centers = None
    elif box_projection_centers[1] is None:
        box_projection_centers[1] = 1 - box_projection_centers[0]

    dataset = MultiCamDataset(
        annotation_paths=annotation_paths,
        image_paths=img_paths,
        calibration_paths=calibration_paths,
        camera_names=cameras,
        ground_truth_paths=ground_truth_paths,
        precomputed=cfg.encoder.name == "precomputed",
        nms_threshold=nms_threshold,
        time_offsets=time_offsets,
        roi_paths=roi_paths,
        bottom=cfg.preprocess.bottom,
        box_projection_centers=box_projection_centers,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )
    return dataloader
