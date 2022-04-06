import json
from functools import partial
from pathlib import Path

import os
import io
import torch
import pickle
from PIL import Image
from typing import Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm

from src.data_tools.perturbation_params_tiered_imagenet import PERTURBATION_PARAMS
from src.data_tools.samplers import AfterCorruptionSampler, BeforeCorruptionSampler
from src.data_tools.transform import TransformLoader
from src.data_tools.utils import get_perturbations


class TieredImageNetC(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        spec_file: str,
        target_transform: Optional[Callable] = None,
    ):
        transform = TransformLoader(image_size).get_composed_transform(aug=False)
        super(TieredImageNetC, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.image_size = image_size
        with open(spec_file, "r") as file:
            split_specs = json.load(file)

        self.root = root
        self.split = split
        self.class_list = split_specs["class_names"]
        self.id_to_class = dict(enumerate(self.class_list))
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}


        self.perturbations, self.id_to_domain = get_perturbations(
            split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

        mode=split

        images_pickle_file = os.path.join(
                                self.root, 'tiered-imagenet', '{}_images_png.pkl'.format(mode)
                                )
        labels_pickle_file = os.path.join(
                                self.root, 'tiered-imagenet','{}_labels.pkl'.format(mode)
                                )

        with open(images_pickle_file, 'rb') as images_file:
            self.images = pickle.load(images_file)
        with open(labels_pickle_file, 'rb') as labels_file:
            self.labels = pickle.load(labels_file)
            self.labels = self.labels['label_specific']

        self.id_to_class = dict(enumerate(self.labels))


    def __len__(self):
        return len(self.images_df)*len(self.perturbations)

    def __getitem__(self, item):
        original_data_index = torch.div(item, len(self.perturbations), rounding_mode="floor")
        perturbation_index = item % len(self.perturbations)

        img = Image.open(io.BytesIO(self.images[original_data_index]))
        
        label = self.labels[original_data_index]

        img = img.resize((self.image_size, self.image_size))
        img = self.perturbations[perturbation_index](img)
        if isinstance(img, np.ndarray):
            img = img.astype(np.uint8)
            img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.float32)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, perturbation_index

    def get_images_and_labels(self):
        """
        Provides image paths and corresponding labels, as expected to define our VisionDataset objects.
        Returns:
            tuple(list(str), list(int): respectively the list of all paths to images belonging in the split defined in
            the input JSON file, and their class ids
        """

        image_names = []
        image_labels = []

        for class_id, class_name in enumerate(self.class_list):
            class_images_paths = [
                str(image_path)
                for image_path in (self.root / class_name).glob("*")
                if image_path.is_file()
            ]
            image_names += class_images_paths
            image_labels += len(class_images_paths) * [class_id]

        return image_names, image_labels

    def get_sampler(self):
        return partial(BeforeCorruptionSampler, self)
