import json
from functools import partial
import pickle

import os
import torch
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
from src.data_tools.utils import get_perturbations, load_image_as_array


class MiniImageNetC(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        spec_file: str,
        target_transform: Optional[Callable] = None,
    ):
        transform = TransformLoader(image_size).get_composed_transform(aug=False)
        super(MiniImageNetC, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.image_size = image_size
        
        with open(spec_file, "r") as file:
            split_specs = json.load(file)
        self.perturbations, self.id_to_domain = get_perturbations(
            split_specs["perturbations"], PERTURBATION_PARAMS, image_size
        )
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

        self.split = split
        if 'val' in split:
            mode = 'validation'
        else:
            mode = split
            
        pickle_file = os.path.join(self.root, 'mini-imagenet-cache-' + mode + '.pkl')
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        print(self.data.keys())

        self.x = self.data["image_data"]
        self.y = np.ones(len(self.x))
        
        def index_classes(items):
            idx = {}
            for i in items:
                if (i not in idx):
                    idx[i] = len(idx)
            return idx

        # TODO Remove index_classes from here
        self.class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = self.class_idx[class_name]

        self.labels = self.y
        self.id_to_class = dict(enumerate(self.labels))
        self.images = self.x

    def __len__(self):
        return len(self.images) * len(self.perturbations)

    def __getitem__(self, item):
        original_data_index = torch.div(item, len(self.perturbations), rounding_mode="floor")
        perturbation_index = item % len(self.perturbations)

        img, label = (
            Image.fromarray(self.images[original_data_index]),
            self.labels[original_data_index],
        )

        img = img.resize((self.image_size, self.image_size))
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
