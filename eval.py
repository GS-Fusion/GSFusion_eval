import os
import json
from pathlib import Path

import cv2
import tyro
import torch
import torchvision.transforms.functional as F
from rich.console import Console
from rich.progress import track
from torchmetrics.functional import mean_squared_error
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


CONSOLE = Console(width=120)


@torch.no_grad()
def rgb_eval(data: Path):
    render_path = data / Path("renders")
    gt_path = data / Path("gt")

    image_list = [f for f in os.listdir(render_path) if f.endswith(".png")]
    image_list = sorted(image_list, key=lambda x: int(x.split(".")[0]))
    num_frames = len(image_list)

    mse = mean_squared_error
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
    lpips = LearnedPerceptualImagePatchSimilarity()

    mse_score_batch = []
    psnr_score_batch = []
    ssim_score_batch = []
    lpips_score_batch = []

    CONSOLE.print(
        f"[bold green]Evaluating a total of {num_frames} rgb frames"
    )

    for i in track(range(0, num_frames)):
        rendered_image = cv2.imread(os.path.join(render_path, image_list[i])) / 255
        rendered_image = F.to_tensor(rendered_image).to(torch.float32).unsqueeze(0)
        gt_image = cv2.imread(os.path.join(gt_path, image_list[i])) / 255
        gt_image = F.to_tensor(gt_image).to(torch.float32).unsqueeze(0)           

        mse_score = mse(rendered_image, gt_image)
        mse_score_batch.append(mse_score)
        psnr_score = psnr(rendered_image, gt_image)
        psnr_score_batch.append(psnr_score)
        ssim_score = ssim(rendered_image, gt_image)
        ssim_score_batch.append(ssim_score)
        lpips_score = lpips(rendered_image, gt_image)
        lpips_score_batch.append(lpips_score)

    mean_scores = {
        "mse": float(torch.stack(mse_score_batch).mean().item()),
        "psnr": float(torch.stack(psnr_score_batch).mean().item()),
        "ssim": float(torch.stack(ssim_score_batch).mean().item()),
        "lpips": float(torch.stack(lpips_score_batch).mean().item()),
    }

    for key, value in mean_scores.items():
        print(f"{key}: {value}")

    with open(os.path.join(data, "metrics.json"), "w") as file:
        print(f"Saving results to {os.path.join(data, 'metrics.json')}")
        json.dump(mean_scores, file, indent=2)


def main(data: Path):
    rgb_eval(data)


if __name__ == "__main__":
    tyro.cli(main)
