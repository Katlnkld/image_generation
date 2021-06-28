import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.2)
from torchvision.utils import make_grid
from IPython.display import clear_output

"""Часть кода для отображения картинок взята из одного из семинарских ноутбуков"""

# Денормировка
def denorm(img_tensors):
    return img_tensors * 0.5 + 0.5

# Отображение картинок
def show_images(images, nmax, title=None, figsize=(20, 15)):
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=nmax).permute(1, 2, 0))

# Отображение батча
def show_batch(dl, nmax=8):
    for real, condition in dl:
        show_images(real, nmax)
        show_images(condition, nmax)
        break

# Отображение батча размером 1 во время обучения
def plot_train(x1, x2, x3, mode):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    if mode:
        fig.suptitle(mode)
    ax1.imshow(denorm(x1.cpu().detach().squeeze().permute(1,2,0).numpy()))
    ax1.axis("off")
    ax1.set_title('Input')
    ax2.imshow(denorm(x2.cpu().detach().squeeze().permute(1,2,0).numpy()))
    ax2.axis("off")
    ax2.set_title('Real')
    ax3.imshow(denorm(x3.cpu().detach().squeeze().permute(1,2,0).numpy()))
    ax3.axis("off")
    ax3.set_title('Generated')
