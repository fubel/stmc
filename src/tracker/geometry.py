import json
import os

import torch


class Projector:
    def __init__(self, calibration_path: str):
        """
        Initialize a Projector object. The projector is used to project points between image and world coordinates.

        Args:
            calibration_path (str): Path to the calibration file (JSON).

        Raises:
            FileNotFoundError: If the calibration file is not found.
            ValueError: If the homography is not found in the calibration file.
        """
        if os.path.exists(calibration_path) is False:
            raise FileNotFoundError(f"Calibration file not found at path: {calibration_path}")
        self.calibration_path = calibration_path

        with open(calibration_path, "r") as f:
            calibration = json.load(f)
            try:
                homography_keys = [
                    "homography",
                    "H",
                    "homography_matrix",
                    "homography matrix",
                ]
                valid_homography_key = set(homography_keys).intersection(set(calibration.keys())).pop()
            except KeyError:
                raise ValueError("Homography not found in calibration file.")
            self._homography = torch.Tensor(calibration[valid_homography_key])
            self._inverse_homography = torch.inverse(self._homography)

    def image_to_world(self, points: torch.Tensor) -> torch.Tensor:
        """Projects image points to world coordinates.

        Args:
            points (torch.Tensor): Image points Nx2.

        Returns:
            torch.Tensor: World points Nx3.
        """
        if points.dim() != 2:
            points = points.view(-1, 2)
        if points.size(1) != 2:
            raise ValueError(f"Expected image points to be of shape (N, 2), but got {points.shape}.")
        return self._homography_image_to_world(points)

    def world_to_image(self, points: torch.Tensor) -> torch.Tensor:
        """Projects world points to image coordinates.

        Args:
            points (torch.Tensor): World points Nx3.

        Returns:
            torch.Tensor: Image points Nx2.
        """
        if points.dim() != 2:
            points = points.view(-1, 3)
        if points.size(1) != 3:
            points = torch.cat([points, torch.ones((points.shape[0], 1))], dim=1)
        return self._homography_world_to_image(points)

    def _homography_image_to_world(self, points: torch.Tensor) -> torch.Tensor:
        points = torch.cat([points, torch.ones((points.shape[0], 1))], dim=1)
        device = points.device
        homography = self._inverse_homography.to(device)
        projected_points = torch.matmul(homography, points.t()).t()
        projected_points = projected_points[:, :2] / projected_points[:, 2].reshape(-1, 1)
        return projected_points

    def _homography_world_to_image(self, points: torch.Tensor) -> torch.Tensor:
        device = points.device
        homography = self._homography.to(device)
        projected_points = torch.matmul(homography, points.t()).t()
        projected_points = projected_points[:, :2] / projected_points[:, 2].reshape(-1, 1)
        return projected_points
