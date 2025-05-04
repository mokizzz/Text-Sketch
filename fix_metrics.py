import math
import pathlib
import sys
import time
import zlib
from argparse import ArgumentParser, Namespace

import lightning as L

# from dataloaders import CLIC, Kodak
import matplotlib.pyplot as plt
import numpy as np

# import libraries
import torch
import tqdm
import yaml
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from loguru import logger
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import adjust_sharpness, to_pil_image

import dataloaders
import models_compressai
import prompt_inversion.open_clip as open_clip
import prompt_inversion.optim_utils as prompt_inv
from annotator.hed import HEDdetector
from annotator.util import HWC3, resize_image
from channel import Channel
from image_utils import convert_pil_to_tensor
from metrics import calculate_all_metrics, calculate_fid

if __name__ == "__main__":
    L.seed_everything(42)
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="set14", type=str)
    args = parser.parse_args()

    logger.add(
        "outputs/{time:YYYY-MM-DD}/{time:HH-mm-ss}.log",
        level="INFO",
    )
    logger.info(f"Fix Metrics. Dataset: {args.dataset}")

    dm = dataloaders.get_dataloader(args)
    dm.setup()

    path_to_recon = f"recon_examples/PICS_clip_ntclam1.0/{args.dataset}_recon/"

    all_metrics = []
    all_origi_images = []
    all_recon_images = []
    channel = Channel()
    for i, x in tqdm.tqdm(enumerate(dm.test_dset)):
        if i >= 100 > 0:
            logger.warning("Stopping after processing 100 images")
            break

        if isinstance(x, dict):
            if "image_path" in x.keys():
                image_path = x["image_path"]
            elif "hr" in x.keys():
                image_path = x["hr"]
            else:
                raise ValueError("Image path not found in data")

            x = Image.open(image_path).convert("RGB")

        w, h = x.size
        xhat = Image.open(path_to_recon + f"{i}_recon.png").convert("RGB")
        xhat = xhat.resize((w, h), Image.LANCZOS)

        all_origi_images.append(x)
        all_recon_images.append(xhat)

        # Calculate metrics
        metrics = calculate_all_metrics(x, xhat)
        all_metrics.append(metrics)
        metrics_log = {k: f"{v:.4g}" for k, v in metrics.items()}
        logger.info(f"Metrics {i:03d}: {metrics_log}")

    # 7. Calculate and log average metrics
    avg_metrics = {
        k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]
    }
    avg_metrics["fid"] = calculate_fid(all_origi_images, all_recon_images)
    for metric_name, value in avg_metrics.items():
        logger.info(f"Average {metric_name}: {value:.6g}")

    logger.info(
        f"Results[{args.dataset}]: "
        + "|".join(
            [
                f"{avg_metrics[k]:.6g}"
                for k in [
                    "psnr",
                    "ssim",
                    "ms_ssim",
                    "lpips",
                    "fid",
                ]
            ]
        )
    )
