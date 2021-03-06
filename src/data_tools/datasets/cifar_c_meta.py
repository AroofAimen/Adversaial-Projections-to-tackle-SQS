from functools import partial
import json
import os
from PIL import Image
import pickle
from loguru import logger
from typing import Any, Callable, Optional

import torch
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

from src.data_tools.perturbation_params_cifar_100 import PERTURBATION_PARAMS
from src.data_tools.samplers import BeforeCorruptionSampler
from src.data_tools.transform import TransformLoader
from src.data_tools.utils import get_perturbations


class CIFAR100CMeta(CIFAR100):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        spec_file: str,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):

        transform = TransformLoader(image_size).get_composed_transform(aug=False)
        self.base_folder = '.'
        super(CIFAR10, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        downloaded_list = self.train_list + self.test_list
        self.split = split
        self._load_meta()

        with open(spec_file, "r") as file:
            self.split_specs = json.load(file)

        self.class_to_idx = {
            class_name: self.class_to_idx[class_name]
            for class_name in self.split_specs["class_names"]
        }
        self.id_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.images: Any = []
        self.labels = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                items_to_keep = [
                    item
                    for item in range(len(entry["data"]))
                    if entry["fine_labels"][item] in self.class_to_idx.values()
                ]
                self.images.append([entry["data"][item] for item in items_to_keep])
                self.labels.extend(
                    [entry["fine_labels"][item] for item in items_to_keep]
                )

        self.images = np.vstack(self.images).reshape(-1, 3, 32, 32)
        self.images = self.images.transpose((0, 2, 3, 1))  # convert to HWC

        self.perturbations, self.id_to_domain = get_perturbations(
            self.split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )

    def __len__(self):
        return len(self.images) * len(self.perturbations)

    def __getitem__(self, item):
        original_data_index =  torch.div(item, len(self.perturbations), rounding_mode="floor")
        perturbation_index = item % len(self.perturbations)


        img, label = (
            Image.fromarray(self.images[original_data_index]),
            self.labels[original_data_index],
        )
        img = self.perturbations[perturbation_index](img)

        if self.transform is not None:
            # TODO: some perturbations output arrays, some output images. We need to clean that.
            if isinstance(img, np.ndarray):
                img = img.astype(np.uint8)
                img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, perturbation_index

    def get_sampler(self):
        return partial(BeforeCorruptionSampler, self)
