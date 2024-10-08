import os

import numpy as np
import torch
from scipy.ndimage import zoom
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, he_dir, srs_dir, scale_factors=[1, 0.8, 0.6, 0.4, 0.2], max_size=None):
        self.he_dir = he_dir
        self.srs_dir = srs_dir
        self.scale_factors = scale_factors
        self.max_size = max_size

        self.he_files = sorted([f for f in os.listdir(he_dir) if f.endswith(".tif")])
        self.srs_files = sorted([f for f in os.listdir(srs_dir) if f.endswith(".tif")])
        assert len(self.he_files) == len(self.srs_files), "Number of files in HE and SRS folders must be same"

    def __len__(self):
        return len(self.he_files) * len(self.scale_factors)

    def __getitem__(self, idx):
        img_idx = idx // len(self.scale_factors)
        scale_idx = idx % len(self.scale_factors)

        he_path = os.path.join(self.he_dir, self.he_files[img_idx])
        srs_path = os.path.join(self.srs_dir, self.srs_files[img_idx])

        he_image = io.imread(he_path)
        srs_image = io.imread(srs_path)

        if self.max_size is not None:
            he_image, srs_image = self.resize(he_image, srs_image)

        scale = self.scale_factors[scale_idx]
        he_image, srs_image = self.apply_scale(he_image, srs_image, scale)

        # Ensure HE and SRS images have the same size, he_image is reference
        srs_image = transform.resize(srs_image, he_image.shape, order=3, anti_aliasing=True)

        he_image, srs_image = self.apply_augmentations(he_image, srs_image)

        he_image = self.to_tensor(he_image)
        srs_image = self.to_tensor(srs_image)

        return he_image, srs_image

    def resize(self, he_image, srs_image):
        """Resize images to have maximum size of max_size if needed."""
        h, w = he_image.shape
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            he_image = transform.resize(he_image, (int(h * scale), int(w * scale)), order=3, anti_aliasing=True)
            srs_image = transform.resize(srs_image, (int(h * scale), int(w * scale)), order=3, anti_aliasing=True)
        return he_image, srs_image

    def apply_scale(self, he_image, srs_image, scale):
        """Apply scale factor to images to increase the versatility of the dataset."""
        if scale != 1.0:
            he_image = zoom(he_image, scale, order=3)
            srs_image = zoom(srs_image, scale, order=3)
        return he_image, srs_image

    def apply_augmentations(self, he_image, srs_image):
        """Apply random augmentations to the images."""
        if np.random.rand() > 0.5:
            he_image = np.fliplr(he_image)
            srs_image = np.fliplr(srs_image)

        if np.random.rand() > 0.5:
            he_image = np.flipud(he_image)
            srs_image = np.flipud(srs_image)

        angle = np.random.randint(-10, 11)
        he_image = transform.rotate(he_image, angle, preserve_range=True)
        srs_image = transform.rotate(srs_image, angle, preserve_range=True)

        brightness_factor = np.random.uniform(0.8, 1.2)
        he_image = np.clip(he_image * brightness_factor, 0, 1)

        return he_image, srs_image

    def to_tensor(self, image):
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        return torch.from_numpy(image).unsqueeze(0)  # Add channel dimension

    def get_img_size(self):
        sample_img = io.imread(os.path.join(self.he_dir, self.he_files[0]))
        return sample_img.shape
