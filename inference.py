import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb
from skimage import io, transform

from model import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inference_args():
    parser = argparse.ArgumentParser(description="Inference LWReg model for image registration")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pth", help="Path to checkpoint")
    parser.add_argument("--he_image", type=str, default="data/he_image/he_2.tif", help="Path to HE image")
    parser.add_argument("--srs_image", type=str, default="data/srs_image/srs_2.tif", help="Path to SRS image")
    parser.add_argument("--output_dir", type=str, default="pred", help="Directory to save results")
    return parser.parse_args()


def infer():
    args = inference_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = LWReg(in_channel=1, out_channel=2, base_channel=8).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    he_image = io.imread(args.he_image)
    srs_image = io.imread(args.srs_image)
    he_image = torch.from_numpy(he_image).unsqueeze(0).unsqueeze(0).to(device)
    srs_image = torch.from_numpy(srs_image).unsqueeze(0).unsqueeze(0).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # Perform inference
    with torch.no_grad():
        flow = model(he_image, srs_image)
        h, w = he_image.shape[-2:]
        stn = SpatialTransform2D((h, w)).to(device)
        warped_srs = stn(srs_image, flow)

    he_np = he_image.squeeze().cpu().numpy()
    srs_np = srs_image.squeeze().cpu().numpy()
    warped_srs_np = warped_srs.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    axes[0, 0].imshow(he_np, cmap="gray")
    axes[0, 0].set_title("Original HE Image")
    axes[0, 1].imshow(srs_np, cmap="gray")
    axes[0, 1].set_title("Original SRS Image")
    axes[0, 2].imshow(warped_srs_np, cmap="gray")
    axes[0, 2].set_title("Warped SRS Image")

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "comparison_result.png"))
    plt.close()


if __name__ == "__main__":
    infer()
