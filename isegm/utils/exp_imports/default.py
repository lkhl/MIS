from functools import partial

import torch
from albumentations import *
from easydict import EasyDict as edict

from isegm.data.datasets import *
from isegm.data.points_sampler import MultiPointSampler
from isegm.data.transforms import *
from isegm.engine.trainer import ISTrainer
from isegm.model import initializer
from isegm.model.is_deeplab_model import DeeplabModel
from isegm.model.is_hrformer_model import HRFormerModel
from isegm.model.is_hrnet_model import HRNetModel
from isegm.model.is_plainvit_model import PlainVitModel
from isegm.model.is_segformer_model import SegformerModel
from isegm.model.is_swinformer_model import SwinformerModel
from isegm.model.losses import *
from isegm.model.metrics import AdaptiveIoU
from isegm.utils.log import logger
