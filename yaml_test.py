
import os
from collections import OrderedDict as ODict

import albumentations
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from albumentations import *
import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset

import argparse
import logging
from typing import Dict, List, Any, Tuple

albumentations_transform = Compose([ShiftScaleRotate(shift_limit=0.11,
                                                     scale_limit=0.1,
                                                     rotate_limit=30,
                                                     interpolation=cv2.INTER_LANCZOS4,
                                                     border_mode=cv2.BORDER_CONSTANT,
                                                     p=0.75),
                                    OneOf([OpticalDistortion(border_mode=cv2.BORDER_CONSTANT,
                                                             p=1.0),
                                           GridDistortion(border_mode=cv2.BORDER_CONSTANT,
                                                          p=1.0)],
                                          p=0.75),
                                    Normalize(mean=[0.1307], std=[0.3081])])

albumentations_valtransform = Compose([Normalize(mean=[0.1307], std=[0.3081])])

albumentations.save(albumentations_transform, "settings.yaml", data_format="yaml")

