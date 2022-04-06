from .cifar_c_meta import CIFAR100CMeta
from .femnist import FEMNIST
from .mini_imagenet_c import MiniImageNetC
from .tiered_imagenet_c import TieredImageNetC


_DATASETS = {
    "mini-imagenet": MiniImageNetC,
    "cifar-100": CIFAR100CMeta,
    "tiered-imagenet": TieredImageNetC,
    "femnist": FEMNIST
}