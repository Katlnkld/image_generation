import os
from torch.utils.data import DataLoader, Dataset
from glob import glob
from pathlib import Path
from PIL import Image
import torchvision.transforms as tt
import re

class LowlightDataset(Dataset):
    def __init__(self, path_d, path_l):
        self.filenames_dark = glob(str(Path(path_d) / "*"))
        self.filenames_light = glob(str(Path(path_l) / "*"))
        self.filenames_dark = sorted(self.filenames_dark,  key=lambda fname: int(re.split(r'[/\.]',fname)[3]))
        self.filenames_light = sorted(self.filenames_light,  key=lambda fname: int(re.split(r'[/\.]',fname)[3]))
    def __len__(self):
        return len(self.filenames_dark)  

    def __getitem__(self, idx):
        transform=tt.Compose([
                tt.ToTensor(),
                tt.Resize((256, 256)),
                tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

        dark = self.filenames_dark[idx]
        light = self.filenames_light[idx]
            
        dark_img = Image.open(dark)
        light_img = Image.open(light)

        dark_img = transform(dark_img)
        light_img = transform(light_img)
        return dark_img, light_img
