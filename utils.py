import cv2
import numpy as np
from pathlib import Path
import torch
from torch import nn
from torch import Tensor
from nerfstudio.utils import colormaps

# Depth Scale Factor m to mm
SCALE_FACTOR = 0.001


def depth_path_to_tensor(
    depth_path: Path, scale_factor: float = SCALE_FACTOR, return_color=False
) -> Tensor:
    """Load depth image in either .npy or .png format and return tensor

    Args:
        depth_path: Path
        scale_factor: float
        return_color: bool
    Returns:
        depth tensor and optionally colored depth tensor
    """
    if depth_path.suffix == ".png":
        depth = cv2.imread(str(depth_path.absolute()), cv2.IMREAD_ANYDEPTH)
    elif depth_path.suffix == ".npy":
        depth = np.load(depth_path, allow_pickle=True)
        if len(depth.shape) == 3:
            depth = depth[..., 0]
    else:
        raise Exception(f"Format is not supported {depth_path.suffix}")
    depth = depth * scale_factor
    depth = depth.astype(np.float32)
    depth = torch.from_numpy(depth).unsqueeze(-1)
    if not return_color:
        return depth
    else:
        depth_color = colormaps.apply_depth_colormap(depth)
        return depth, depth_color  # type: ignore


class DepthMetrics(nn.Module):
    """Computation of error metrics between predicted and ground truth depths

    from:
        https://arxiv.org/abs/1806.01260

    Returns:
        abs_rel: normalized avg absolute realtive error
        sqrt_rel: normalized square-root of absolute error
        rmse: root mean square error
        rmse_log: root mean square error in log space
        a1, a2, a3: metrics
    """

    def __init__(self, tolerance: float = 0.1, **kwargs):
        self.tolerance = tolerance
        super().__init__()

    @torch.no_grad()
    def forward(self, pred, gt):
        mask = gt > self.tolerance

        thresh = torch.max((gt[mask] / pred[mask]), (pred[mask] / gt[mask]))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25**2).float().mean()
        a3 = (thresh < 1.25**3).float().mean()
        rmse = (gt[mask] - pred[mask]) ** 2
        rmse = torch.sqrt(rmse.mean())

        rmse_log = (torch.log(gt[mask]) - torch.log(pred[mask])) ** 2
        # rmse_log[rmse_log == float("inf")] = float("nan")
        rmse_log = torch.sqrt(rmse_log).nanmean()

        abs_rel = torch.abs(gt - pred)[mask] / gt[mask]
        abs_rel = abs_rel.mean()
        sq_rel = (gt - pred)[mask] ** 2 / gt[mask]
        sq_rel = sq_rel.mean()

        return (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)