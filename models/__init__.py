# coding: utf-8

import sys
from .dla import *
from .resnet import *
from .generic_efficientnet import *
from .utils.helpers import *
from .sync_batchnorm import *

mod = sys.modules[__name__]

def create_model(model_name, num_classes, pretrained, in_chans, **kwargs):
	return getattr(mod, model_name)(num_classes=num_classes, pretrained=pretrained, in_chans=in_chans, **kwargs)