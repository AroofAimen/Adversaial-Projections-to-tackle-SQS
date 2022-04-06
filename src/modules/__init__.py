from .optimal_transport import OptimalTransport
from .backbones import *
from .resnet import resnet12, resnet18

_BACKBONES = {
    "conv4": Conv4,
    "resnet12": resnet12,
    "resnet18": resnet18
}

def get_backbone(config):
    """
    Get backbone from config
    """
    assert config.backbone in _BACKBONES, f"Backbone {config.backbone} not found"
    return _BACKBONES[config.backbone](**{
        "image_size": config.image_size,
        "use_elu": config.use_elu
        })
    