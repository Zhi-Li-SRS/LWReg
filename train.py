import argparse
import logging
import os
import random
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import hsv_to_rgb
from skimage import io
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageDataset
from losses import *
from model import *


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_args():
    parser = argparse.ArgumentParser(description="Train LWReg model for image registration")
    parser.add_argument("--he_dir", type=str, default="data/he_image", help="Directory containing HE images")
    parser.add_argument("--srs_dir", type=str, default="data/srs_image", help="Directory containing SRS images")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--smooth_weight", type=float, default=0.0001, help="Weight for smoothness loss")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--patch_size", type=int, default=256, help="Size of image patches")
    parser.add_argument(
        "--scale_factors",
        nargs="+",
        type=float,
        default=[1.0, 0.8, 0.6, 0.4, 0.2],
        help="Scale factors for multi-scale training",
    )
    parser.add_argument("--resume", type=bool, default=False, help="Resume training from the last checkpoint")

    return parser.parse_args()


def save_checkpoint(state, is_best, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        torch.save(state, best_path)


def setup_logger(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time_stamp = datetime.now().strftime("%H-%M-%S")
    log_filename = f"train_e{args.num_epochs}_lr{args.learning_rate}_{time_stamp}.log"
    log_path = os.path.join(args.log_dir, log_filename)
    logging.basicConfig(filename=log_path, level=logging.INFO)

    return logging.getLogger()


def plot_loss(epochs, losses, log_dir, num_epochs, learning_rate):
    plt.figure(figsize=(10, 5))
    plt.style.use("ggplot")
    plt.plot(epochs, losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    timestamp = datetime.now().strftime("%H%M%S")
    plot_filename = f"loss_e{num_epochs}_lr{learning_rate}_{timestamp}.png"
    plot_path = os.path.join(log_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()


def visualize_flow(flow, epoch, save_dir):
    flow = flow.squeeze().cpu().numpy()

    # Calculate flow magnitude and angle
    magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    angle = np.arctan2(flow[1], flow[0])
    magnitude = magnitude / magnitude.max()

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
    hsv[..., 0] = (angle + np.pi) / (2 * np.pi)  # Hue
    hsv[..., 1] = magnitude  # Saturation
    hsv[..., 2] = 1  # Value

    rgb = hsv_to_rgb(hsv)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)

    # Add arrows to show flow direction
    step = 20  # Adjust this to change arrow density
    y, x = np.mgrid[step // 2 : flow.shape[1] : step, step // 2 : flow.shape[2] : step]
    fx = flow[0, y, x]
    fy = flow[1, y, x]
    ax.quiver(x, y, fx, fy, color="w", angles="xy", scale_units="xy", scale=0.1)

    ax.set_title(f"Flow Visualization - Epoch {epoch}")
    ax.axis("off")

    ax_cb = fig.add_axes([0.9, 0.1, 0.03, 0.8])  # [left, bottom, width, height]
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap="hsv"), cax=ax_cb)
    cb.set_ticks([])
    cb.set_label("Flow Direction", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"flow_e{epoch+1}.png"), dpi=300, bbox_inches="tight")
    plt.close()


def trainer():
    args = train_args()
    set_seed(3407)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(args)
    logger.info(f"Using device: {device}")

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    dataset = ImageDataset(args.he_dir, args.srs_dir, scale_factors=args.scale_factors)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = LWReg(in_channel=1, out_channel=2, base_channel=8).to(device)
    criterion = mi_loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    start_epoch = 0
    best_loss = float("inf")
    losses = []
    epochs = []

    # Load checkpoint if resuming training
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
        logger.info(f"Resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss = 0.0

        for he, srs in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            he, srs = he.to(device), srs.to(device)

            # Get the current height and width of the image
            h, w = he.size()[-2:]
            stn = SpatialTransform2D(size=(h, w)).to(device)
            diff = DiffeomorphicTransform2D(size=(h, w)).to(device)

            optimizer.zero_grad()

            flow = model(he, srs)
            warped_srs = stn(srs, flow)
            flow_backward = model(warped_srs, he)
            recon_he = stn(warped_srs, flow_backward)

            loss_similarity = criterion(he, warped_srs)
            loss_cycle = F.mse_loss(he, recon_he)

            loss_smooth = smoothloss(flow) + smoothloss(flow_backward)
            loss = loss_similarity + args.smooth_weight * loss_smooth + 0.1 * loss_cycle

            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss}")
        losses.append(avg_loss)
        epochs.append(epoch)

        scheduler.step()

        is_best = avg_loss < best_loss
        best_loss = min(avg_loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.checkpoint_dir,
        )

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                # only process the first image in the dataset
                he, srs = dataset[0]
                he, srs = he.to(device), srs.to(device)
                flow = model(he.unsqueeze(0), srs.unsqueeze(0))
                # visualize_flow(flow, epoch, args.results_dir)
                warped_srs = stn(srs.unsqueeze(0), flow)

                he_np = he.squeeze().cpu().numpy()
                srs_np = srs.squeeze().cpu().numpy()
                warped_srs_np = warped_srs.squeeze().cpu().numpy()
                diff_image = np.abs(he_np - warped_srs_np)

                fig, axes = plt.subplots(2, 2, figsize=(20, 20))
                axes[0, 0].imshow(he_np, cmap="gray")
                axes[0, 0].set_title("HE Image")
                axes[0, 1].imshow(srs_np, cmap="gray")
                axes[0, 1].set_title("SRS Image")
                axes[1, 0].imshow(warped_srs_np, cmap="gray")
                axes[1, 0].set_title("Warped SRS Image")
                axes[1, 1].imshow(diff_image, cmap="hot")
                axes[1, 1].set_title("Difference Image")

                plt.tight_layout()
                plt.savefig(os.path.join(args.results_dir, f"comparison_e{epoch+1}.png"))
                plt.close()
            model.train()
        # Plot and save the loss curve
    plot_loss(epochs, losses, args.log_dir, args.num_epochs, args.learning_rate)
    logger.info(f"Training completed. Loss plot saved in {args.log_dir}")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument."
    )
    trainer()
