from .matchingnet import MatchingNet
from .protonet import ProtoNet
from .transductive_propagation_net import TransPropNet
from .uda_ot import UnsupDomAdapOT
from .maximum_a_posteriori import MAP


_METHODS = {
    "MatchingNet": MatchingNet,
    "ProtoNet": ProtoNet,
    "TransPropNet": TransPropNet,
}

def get_model(method):
    """
    Returns the model class.
    """
    assert method in _METHODS, f"Method {method} not found."
    return _METHODS[method]