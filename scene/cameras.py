"""
Camera objects used by rendering and training.

Camera loads image tensors, alpha masks, and camera
matrices. MiniCam is a smaller camera view for code paths that only need
projection data.
"""
from __future__ import annotations

import torch
from torch import nn
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch




def _resolve_camera_device(data_device: str) -> torch.device:
    """
        Convert a device string to torch.device, falling back to CUDA on failure.
    """
    try:
        return torch.device(data_device)
    except Exception as e:
        print(e)
        print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
        return torch.device("cuda")


def _build_alpha_mask(
    resized_image_uint8: torch.Tensor,
    data_device: torch.device,
) -> torch.Tensor | None:
    """
        Build the alpha mask used to ignore invalid image regions.

        RGBA images provide the mask; RGB images have no explicit mask.
    """
    if resized_image_uint8.shape[0] != 4:
        return None
    return resized_image_uint8[3:4, ...].to(data_device)


class Camera(nn.Module):
    """
        Full camera record with image data and projection matrices.
    """

    def __init__(
        self,
        resolution: tuple[int, int],
        colmap_id: int,
        R: NDArray[np.floating],
        T: NDArray[np.floating],
        FoVx: float,
        FoVy: float,
        image: PILImage.Image,
        image_name: str,
        uid: int,
        trans: NDArray[np.floating] = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
    ) -> None:
        """
            Load image tensors and precompute transforms for rendering.
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.data_device = _resolve_camera_device(data_device)

        resized_image_uint8 = PILtoTorch(image, resolution)
        gt_image_uint8 = resized_image_uint8[:3, ...]
        alpha_mask = _build_alpha_mask(
            resized_image_uint8,
            self.data_device,
        )

        self._alpha_mask_uint8 = alpha_mask
        self._original_image_uint8 = gt_image_uint8.to(self.data_device)
        self.image_width = self._original_image_uint8.shape[2]
        self.image_height = self._original_image_uint8.shape[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    @property
    def original_image(self) -> torch.Tensor:
        return self._original_image_uint8.float() / 255.0

    def original_image_uint8_cuda(self) -> torch.Tensor:
        return self._original_image_uint8.to("cuda", non_blocking=True)

    def original_image_cuda(self) -> torch.Tensor:
        return self.original_image_uint8_cuda().float().div_(255.0)

    @property
    def alpha_mask(self) -> torch.Tensor | None:
        if self._alpha_mask_uint8 is None:
            return None
        return self._alpha_mask_uint8.float() / 255.0

    def alpha_mask_cuda(self) -> torch.Tensor | None:
        if self._alpha_mask_uint8 is None:
            return None
        return self._alpha_mask_uint8.to("cuda", non_blocking=True).float().div_(255.0)

class MiniCam:
    """
        Lightweight camera view that stores only projection-related fields.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fovy: float,
        fovx: float,
        znear: float,
        zfar: float,
        world_view_transform: torch.Tensor,
        full_proj_transform: torch.Tensor,
    ) -> None:
        """
            Store projection fields and derive the camera center from the view matrix.
        """
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
