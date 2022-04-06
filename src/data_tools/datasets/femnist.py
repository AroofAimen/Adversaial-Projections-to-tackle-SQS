from functools import partial
from pathlib import Path
import os

from typing import Callable, Optional

import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision import transforms

from src.data_tools.samplers import GroupedDatasetSampler


class FEMNIST(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str,
        image_size: int,
        spec_file: str,
        target_transform: Optional[Callable] = None,
    ):
        transform = transforms.ToTensor()

        super(FEMNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.root = root

        self.images = np.load(os.path.join(self.root, f"{split}.npy"))

        self.meta_data = pd.read_csv(os.path.join(spec_file), index_col=0)

        self.id_to_class = dict(
            enumerate(self.meta_data.class_name.sort_values().unique())
        )
        self.class_to_id = {v: k for k, v in self.id_to_class.items()}
        self.id_to_domain = dict(enumerate(self.meta_data.user.unique()))
        self.domain_to_id = {v: k for k, v in self.id_to_domain.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        label = self.class_to_id[self.meta_data.class_name.iloc[int(item)]]
        domain_id = self.domain_to_id[self.meta_data.user.iloc[int(item)]]
        img = self.transform(self.images[int(item)]).repeat(3, 1, 1).float()

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, domain_id

    def get_sampler(self):
        return partial(GroupedDatasetSampler, self)
