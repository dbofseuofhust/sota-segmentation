from .base import *
from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .cityscapes import CityscapesSegmentation
from .cityscapes_oc import CitySegmentationTrain,CitySegmentationTest,CitySegmentationTrainWpath
from .ead import EADSegmentation
from .crack import CrackSegmentation
from .monusac import MonusacSegmentation
from .disease import DiseaseSegmentation
from .buildings import BuildingSegmentation,BuildingSegmentation2


datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'cityscapes': CityscapesSegmentation,
    'cityscapes_oc': CitySegmentationTrain,
    'ead': EADSegmentation,
    'crack': CrackSegmentation,
    'monusac': MonusacSegmentation,
    'disease': DiseaseSegmentation,
    'buildings': BuildingSegmentation,
    'buildings2': BuildingSegmentation2,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)

ocdatasets = {
	'cityscapes_train': CitySegmentationTrain,
	'cityscapes_test': CitySegmentationTest,
	'cityscapes_train_w_path': CitySegmentationTrainWpath,
}


def get_ocsegmentation_dataset(name, **kwargs):
    return ocdatasets[name.lower()](**kwargs)
