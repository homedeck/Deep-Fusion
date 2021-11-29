# 
#   Deep Fusion
#   Copyright (c) 2020 Homedeck, LLC.
#

from io import BytesIO
from PIL import Image
from requests import get
from suya import Suya
from suya.torch.cache import TensorCache
from torch import cat
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

class FusionDataset (Dataset):
    
    def __init__ (self, tag, dataset_size=1000, patch_size=512):        
        self.__sampler, self.__size = Suya().paired_dataset(tag, size=dataset_size)
        self.__cache = TensorCache()
        self.__transform = Compose([
            Resize(patch_size),
            CenterCrop(patch_size),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__ (self):
        return self.__size

    def __getitem__ (self, index):
        # Get sample
        exposures, fusion = self.__sampler(index)
        exposures = self.__cache.fetch(*exposures, transform=self.__transform)
        fusion = self.__cache.fetch(fusion, transform=self.__transform)
        # Stack exposures
        exposure_stack = cat(exposures, dim=0)
        return exposure_stack, fusion

