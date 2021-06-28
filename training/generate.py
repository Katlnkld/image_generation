import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as tt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.2)
from torchvision.utils import make_grid
from IPython.display import clear_output

pix2pix = {
    "discriminator": Discriminator().to(device),
    "generator": Generator().to(device)
}

criterion = {
    "discriminator": nn.BCELoss(),
    "generator": nn.L1Loss()
}

# Загрузка весов модели
checkpoint = torch.load(path)

pix2pix['discriminator'].load_state_dict(checkpoint['model_discriminator_state_dict'])
pix2pix['generator'].load_state_dict(checkpoint['model_generator_state_dict'])

# Генерация
for x, y in valloader:
    pix2pix["generator"].eval()
    with torch.no_grad():
        fake_y = pix2pix['generator'](x.to(device))
        show_images(x.cpu(), 6, figsize=(20, 4), title='Input')
        show_images(y.cpu(), 6, figsize=(20, 4), title='Ground truth')
        show_images(fake_y.cpu(), 6, figsize=(20, 4), title='Generated')
