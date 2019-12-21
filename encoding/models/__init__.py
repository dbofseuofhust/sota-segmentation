from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .oc_module import *
from .psp import *
from .encnet import *
from .danet import *
from .resnet101_asp_oc import get_resnet101_asp_oc_dsn
from .resnet101_base_oc import get_resnet101_base_oc_dsn
from .resnet101_baseline import get_resnet101_baseline
from .resnet101_pyramid_oc import get_resnet101_pyramid_oc_dsn
from .emanet import get_emanet
from .galdnet import get_galdnet
from .deeplabv3 import get_deeplabv3
from .ccnet import get_ccnet

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'psp': get_psp,
        'encnet': get_encnet,
        'danet': get_danet,
        'asp_oc_dsn': get_resnet101_asp_oc_dsn,
        'base_oc_dsn': get_resnet101_base_oc_dsn,
        'pyramid_oc_dsn': get_resnet101_pyramid_oc_dsn,
        'emanet': get_emanet,
        'galdnet': get_galdnet,
        'deeplabv3': get_deeplabv3,
        'ccnet': get_ccnet,
    }
    return models[name.lower()](**kwargs)

networks = {
    'resnet101_baseline': get_resnet101_baseline,
    'resnet101_base_oc_dsn': get_resnet101_base_oc_dsn,
    'resnet101_pyramid_oc_dsn': get_resnet101_pyramid_oc_dsn,
    'resnet101_asp_oc_dsn': get_resnet101_asp_oc_dsn,
}

def get_ocsegmentation_model(name, **kwargs):
    return networks[name.lower()](**kwargs)
