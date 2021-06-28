import os
from torch.utils.data import DataLoader, Dataset
from glob import glob
from pathlib import Path
from PIL import Image
import torchvision.transforms as tt

"""
Часть кода для загрузки датасета взята [отсюда](https://colab.research.google.com/github/LibreCV/blog/blob/master/_notebooks/2021-02-13-Pix2Pix%20explained%20with%20code.ipynb?authuser=2).
"""

class FacadesDataset(Dataset):
    def __init__(self, path):
        self.filenames = glob(str(Path(path) / "*"))
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        transform=tt.Compose([
                tt.Resize(256),
                tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        filename = self.filenames[idx]
        image = Image.open(filename)
        image = tt.functional.to_tensor(image)
        image_width = image.shape[2]

        real = image[:, :, : image_width // 2]
        real = transform(real)
        condition = image[:, :, image_width // 2 :]
        condition = transform(condition)

        return condition, real