import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import expand_boxes, remove_border_boxes, size_filter


class ResultsWriter:
    def __init__(self, output_path, cfg, normalization=None, camera_names=None):
        self._results = []

        self.cfg = cfg
        self.output_path = output_path
        self._norm_factors = normalization

        self.rows = cfg.visuals.grid_rows
        self.plot_results = cfg.visuals.plot_results
        self.plot_every = cfg.visuals.plot_interval

        self.camera_names = camera_names

        self.writer = None

        if cfg.logging.tensorboard.enable:
            self.writer = SummaryWriter()

        self.store_files = cfg.visuals.store_files
        self.results_file = os.path.join(output_path, "results.txt")

        self.offsets = cfg.dataset.offsets if hasattr(cfg.dataset, "offsets") else [0] * len(camera_names)

        self.on_bev = True if cfg.dataset.name == "WildTrack" else False

        self._save_function = self.get_save_function(cfg)

        if os.path.exists(self.results_file):
            os.remove(self.results_file)

        os.makedirs(output_path, exist_ok=True)

    @property
    def results(self):
        results = torch.cat(self._results, dim=0)
        for i, offset in enumerate(self.offsets):
            results[results[:, 0] == i, 2] -= offset
        # multiply camera column by (-1)
        results[:, 0] *= -1
        for i, name in enumerate(self.camera_names):
            # this is a bit hacky if camera does not start with letter
            try:
                name_int = int(name[1:])
            except ValueError:
                # fallback to index of camera
                name_int = i
            results[results[:, 0] == -i, 0] = name_int
        if self.cfg.postprocess.expand_boxes.enable:
            factor = self.cfg.postprocess.expand_boxes.factor
            results[:, 3:7] = expand_boxes(results[:, 3:7], factor)
        if self.cfg.postprocess.remove_borders.enable:
            boxes = results[:, 3:7]
            border = self.cfg.postprocess.remove_borders.border_size
            keep = remove_border_boxes(boxes, border)
            results = results[keep]
        if self.cfg.postprocess.size_filter.enable:
            boxes = results[:, 3:7]
            keep = size_filter(
                boxes, self.cfg.postprocess.size_filter.min_size, self.cfg.postprocess.size_filter.max_size
            )
            results = results[keep]
        return results

    def add(self, result):
        _result = result.clone()
        if self._norm_factors is not None:
            _result = self.denormalize_bev(_result[:, 7:9])
        self._results.append(result)

    def save(self):
        if self._results:
            self._save_function(self.results.cpu().numpy())

    def _to_aicity19(self, result):
        # CAMERA_ID  OBJ_ID  FRAME  X Y  W  H  1  X_BEV  Y_BEV  -1
        np.savetxt(self.results_file, result, fmt="%d %d %d %d %d %d %d %f %f")

    def _to_aicity24(self, result):
        # CAMERA_ID  OBJ_ID FRAME  X Y  W  H  1  X_BEV  Y_BEV  -1
        np.savetxt(self.results_file, result, fmt="%d %d %d %d %d %d %d %f %f")

    def _to_synthehicle(self, result):
        # CAMERA, FRAME, ID, X, Y, W, H, SCORE, X_BEV, Y_BEV
        np.savetxt(self.results_file, result[:, [2, 1]], fmt="%d", delimiter=",")

    def get_save_function(self, cfg):
        if "WildTrack" in cfg.dataset.name:
            return self._to_wildtrack
        elif "AICITY24" in cfg.dataset.name:
            return self._to_aicity19
        elif "AICITY" in cfg.dataset.name or "CityFlow" in cfg.dataset.name:
            return self._to_aicity24
        else:
            return self._to_synthehicle

    def denormalize_bev(self, positions):
        min_x, min_y, max_x, max_y = self._norm_factors
        return positions * torch.tensor([max_x - min_x, max_y - min_y]) + torch.tensor([min_x, min_y])

    def squeeze_batch(self, x: torch.Tensor):
        if x.dim() == 4 and x.size(0) == 1:
            return x.squeeze(0)
        return x
