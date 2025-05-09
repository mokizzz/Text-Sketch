"""This script runs the full pipeline
for Prompt-Inversion Compression w/ Sketch"""

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

# global variables to enhance input text prompts in stable diffusion model
prompt_pos = "high quality"
prompt_neg = "disfigured, deformed, low quality, lowres, b&w, blurry, Photoshop, video game, bad art"


def get_loss(loss_type: str):
    """
    Helper function to define loss in Reverse Channel Coding scheme

    Arguments:
        loss_type: type of loss to use (str)

    Returns:
        loss_function: loss function used to determine best index in RCC (function)
    """
    if loss_type == "clip":
        # create Namespace() for clip model args
        args_clip = Namespace()
        # populate args_clip with entries from config file
        args_clip.__dict__.update(
            prompt_inv.read_json("prompt_inversion/sample_config.json")
        )
        # instantiate pretrained clip model
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            args_clip.clip_model, pretrained=args_clip.clip_pretrain, device="cuda:0"
        )
        # return loss function that computes cosine similarity between input images
        return lambda x, xhat: 1 - prompt_inv.clip_cosine(
            x, xhat, clip_model, clip_preprocess, "cuda:0"
        )
    else:
        sys.exit("Not a valid loss")


def encode_rcc(model, clip, preprocess, ntc_sketch, im, N=5):
    """
    Function to encode image using prompt inversion and generate HED sketch on the side

    Arguments:
        model: ControlNet model
        clip: CLIP model
        preprocess: CLIP model preprocess
        ntc_sketch: NTC model
        im: input image to compress
        N: number of candidate images to generate

    Returns:
        caption: text string containing caption (str)
        sketch: HED sketch of original image
        sketch_dict: dict containing compressed sketch
        idx: index selected (int)
    """
    # generate hed map and preprocess to match required input to NTC encoder
    apply_hed = HEDdetector()
    hed_map = HWC3(apply_hed(im))
    sketch = Image.fromarray(hed_map)
    sketch = ntc_preprocess(sketch).unsqueeze(0)

    # compress sketch using NTC encoder
    # reconstruct sketch using NTC encoder to generate candidate images in RCC
    with torch.no_grad():
        sketch_dict = ntc_sketch.compress(sketch)
        sketch_recon = ntc_sketch.decompress(
            sketch_dict["strings"], sketch_dict["shape"]
        )["x_hat"][0]
        sketch_recon = adjust_sharpness(sketch_recon, 2)
        sketch_recon = HWC3(
            (255 * sketch_recon.permute(1, 2, 0)).numpy().astype(np.uint8)
        )

    # Generate image caption using Prompt Inversion
    # if image has previously been captioned, load saved caption to save time
    try:
        with open(
            f"recon_examples/PICS_clip_ntclam1.0/CLIC2021_recon/{i}_caption.yaml", "r"
        ) as file:
            caption_dict = yaml.safe_load(file)
        caption = caption_dict["caption"]
    except Exception:
        caption = prompt_inv.optimize_prompt(
            clip, preprocess, args_clip, "cuda:0", target_images=[Image.fromarray(im)]
        )

    # run ControlNet model to generate N candidate images
    guidance_scale = 9
    num_inference_steps = 25
    images = model(
        f"{caption}, {prompt_pos}",
        Image.fromarray(sketch_recon),
        generator=[torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
    ).images
    # compute CLIP cosine similarity between original image and generated image candidates
    loss = loss_func([Image.fromarray(im)] * N, images).squeeze()
    # compute index of candidate image that minimizes loss
    idx = torch.argmin(loss)

    return caption, sketch, sketch_dict, idx


def recon_rcc(model, ntc_sketch, caption, sketch_dict, idx, N=5):
    """
    Function to decode image using ControlNet to generate new image using encoded prompt and sketch

    Arguments:
        model: ControlNet model
        ntc_sketch: NTC model
        caption: text string caption
        sketch_dict: dictionary containing compressed sketch
        idx: index of best candidate image
        N: number of candidate images to generate

    Returns:
        im_recon: reconstructed image generated from ControlNet
        sketch_recon: reconstructed sketch from NTC model
    """
    # decode sketch using NTC model
    with torch.no_grad():
        sketch_recon = ntc_sketch.decompress(
            sketch_dict["strings"], sketch_dict["shape"]
        )["x_hat"][0]
        sketch_recon = adjust_sharpness(sketch_recon, 2)
    sketch_recon = HWC3((255 * sketch_recon.permute(1, 2, 0)).numpy().astype(np.uint8))

    # decode image
    guidance_scale = 9
    num_inference_steps = 25
    images = model(
        f"{caption}, {prompt_pos}",
        Image.fromarray(sketch_recon),
        generator=[torch.Generator(device="cuda").manual_seed(i) for i in range(N)],
        num_images_per_prompt=N,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=im.shape[0],
        width=im.shape[1],
        negative_prompt=prompt_neg,
    ).images

    return images[idx], sketch_recon


def ntc_preprocess(image):
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
    image = transform(image)
    return image


if __name__ == "__main__":
    L.seed_everything(42)

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--N", default=4, type=int)
    parser.add_argument("--dataset", default="flickr8k", type=str)
    parser.add_argument("--data_root", default="/home/zhaozr/datasets", type=str)
    parser.add_argument("--loss", default="clip", type=str)
    parser.add_argument("--lam_sketch", default=1.0, type=str)
    parser.add_argument("--snr", default=None, type=float)
    args = parser.parse_args()

    # init logger
    logger.add(
        "outputs/{time:YYYY-MM-DD}/{time:HH-mm-ss}.log",
        level="INFO",
    )
    logger.info(f"Dataset: {args.dataset}")

    # get dataloader
    dm = dataloaders.get_dataloader(args)
    dm.setup()

    # Load ControlNet model for generative decoder
    sd_model_id = "stabilityai/stable-diffusion-2-1-base"
    cn_model_id = "thibaud/controlnet-sd21-hed-diffusers"
    # cn_model_id = "lllyasviel/sd-controlnet-hed"
    controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=torch.float16)
    model = StableDiffusionControlNetPipeline.from_pretrained(
        sd_model_id, controlnet=controlnet, torch_dtype=torch.float16, revision="fp16"
    )
    # set ControlNet configs
    model.scheduler = UniPCMultistepScheduler.from_config(model.scheduler.config)
    model.enable_xformers_memory_efficient_attention()
    model.enable_model_cpu_offload()

    # Load loss function
    loss_func = get_loss(args.loss)

    # Load CLIP model for Prompt Inversion
    args_clip = Namespace()
    args_clip.__dict__.update(
        prompt_inv.read_json("./prompt_inversion/sample_config.json")
    )
    clip, _, clip_preprocess = open_clip.create_model_and_transforms(
        args_clip.clip_model, pretrained=args_clip.clip_pretrain, device="cuda:0"
    )

    # Load NTC model
    args_ntc = Namespace()
    args_ntc.model_name = "Cheng2020AttentionFull"
    args_ntc.lmbda = args.lam_sketch
    args_ntc.dist_name_model = "ms_ssim"
    args_ntc.orig_channels = 1
    ntc_sketch = models_compressai.get_models(args_ntc)
    saved = torch.load(
        f"./models_ntc/{args_ntc.model_name}_CLIC_HED_{args_ntc.dist_name_model}_lmbda{args_ntc.lmbda}.pt"
    )
    ntc_sketch.load_state_dict(saved)
    ntc_sketch.eval()
    ntc_sketch.update()

    # Make savedir
    save_dir = (
        f"./recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}/{args.dataset}_recon"
    )
    sketch_dir = f"./recon_examples/PICS_{args.loss}_ntclam{args_ntc.lmbda}/{args.dataset}_sketch"
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(sketch_dir).mkdir(parents=True, exist_ok=True)

    # iterate through images in dataset and run PICS
    all_metrics = []
    all_origi_images = []
    all_recon_images = []
    channel = Channel(snr=args.snr)
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
            x = convert_pil_to_tensor(x)

        # process image from dataloader
        x = x[0]
        x_im = (255 * x.permute(1, 2, 0)).numpy().astype(np.uint8)
        im = resize_image(HWC3(x_im), 512)
        im_orig = Image.fromarray(im)

        # Encode and decode
        start_time = time.time()
        caption, sketch, sketch_dict, idx = encode_rcc(
            model, clip, clip_preprocess, ntc_sketch, im, args.N
        )
        elapsed_transmit_time = time.time() - start_time

        orig_size = channel.calculate_size_KB(im_orig)
        (caption, sketch_dict, idx), trans_sizes = channel([caption, sketch_dict, idx])

        start_time = time.time()
        xhat, sketch_recon = recon_rcc(
            model, ntc_sketch, caption, sketch_dict, idx, args.N
        )
        elapsed_receive_time = time.time() - start_time

        all_origi_images.append(im_orig)
        all_recon_images.append(xhat)

        # Save ground-truth image
        im_orig.save(f"{sketch_dir}/{i}_gt.png")

        # Save reconstructions
        xhat.save(f"{save_dir}/{i}_recon.png")

        # Save sketch images
        im_sketch = to_pil_image(sketch[0])
        im_sketch.save(f"{sketch_dir}/{i}_sketch.png")

        im_sketch_recon = Image.fromarray(sketch_recon)
        im_sketch_recon.save(f"{sketch_dir}/{i}_sketch_recon.png")

        # Compute rates
        bpp_sketch = sum(
            [
                len(bin(int.from_bytes(s, sys.byteorder)))
                for s_batch in sketch_dict["strings"]
                for s in s_batch
            ]
        ) / (im_orig.size[0] * im_orig.size[1])
        bpp_caption = (
            sys.getsizeof(zlib.compress(caption.encode()))
            * 8
            / (im_orig.size[0] * im_orig.size[1])
        )

        # save results
        compressed = {
            "caption": caption,
            "prior_strings": sketch_dict["strings"][0][0],
            "hyper_strings": sketch_dict["strings"][1][0],
            "bpp_sketch": bpp_sketch,
            "bpp_caption": bpp_caption,
            "bpp_total": bpp_sketch
            + bpp_caption
            + math.log2(args.N) / (im_orig.size[0] * im_orig.size[1]),
        }
        with open(f"{save_dir}/{i}_caption.yaml", "w") as file:
            yaml.dump(compressed, file)

        # Calculate metrics
        metrics = calculate_all_metrics(im_orig, xhat)
        metrics.update(
            {
                "elapsed_transmit_time": elapsed_transmit_time,
                "elapsed_receive_time": elapsed_receive_time,
                "compression_ratio": trans_sizes / orig_size,
            }
        )
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
                    "compression_ratio",
                ]
            ]
        )
    )
