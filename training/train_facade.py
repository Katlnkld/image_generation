import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, Dataset
from glob import glob
from pathlib import Path
from PIL import Image
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
from tqdm.notebook import tqdm
from torchsummary import summary
import re

# Гиперпараметры
path_train = "./facades/train/"
path_val = "./facades/val/"
batch_size = 1
path = '/content/drive/MyDrive/pix2pix_facades/my_p2p_model_facade.pth'
Lambda = 100
epochs = 200
lr = 0.0002

train_dataset = FacadesDataset(path_train)
trainloader_facade = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = FacadesDataset(path_val)
valloader_facade = DataLoader(val_dataset, batch_size=10, shuffle=False)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device

trainloader_facade = DeviceDataLoader(trainloader_facade, device)
valloader_facade = DeviceDataLoader(valloader_facade, device)

# Cоздание модели
pix2pix = {
    "discriminator": Discriminator().to(device),
    "generator": Generator().to(device)
}

criterion = {
    "discriminator": nn.BCELoss(),
    "generator": nn.L1Loss()
}

def train(model, trainloader, valloader, epochs=epochs):
    torch.cuda.empty_cache()
    
    losses_g_train = []
    losses_d_train = []
    
    # Оптимайзеры
    optimizer = {
        "discriminator": torch.optim.Adam(model["discriminator"].parameters(), 
                                          lr=lr, betas=(0.5, 0.999)),
        "generator": torch.optim.Adam(model["generator"].parameters(),
                                      lr=lr, betas=(0.5, 0.999))
    }
    
    for epoch in range(epochs):
        model["discriminator"].train()
        model["generator"].train()
        
        loss_d_per_epoch = []
        loss_g_per_epoch = []

        for inp, out in tqdm(trainloader):
            
            # Обучение генератора
            optimizer["generator"].zero_grad()
            fake = model['generator'](inp)
            preds = model["discriminator"](fake, inp)
            ones = torch.ones(preds.size(0), 1,preds.shape[2],preds.shape[3], device=device)
            
            loss_gan = criterion['discriminator'](preds, ones)
            loss_pixel = criterion['generator'](fake, out)
            loss_g = loss_gan + Lambda*loss_pixel
         
            loss_g.backward()
            optimizer["generator"].step()
            loss_g_per_epoch.append(loss_g.item())

            # Обучение дискриминатора
            optimizer["discriminator"].zero_grad()
            real_preds = model["discriminator"](out, inp)
            ones = torch.ones(real_preds.size(0), 1,real_preds.shape[2],real_preds.shape[3], device=device)
            real_loss = criterion['discriminator'](real_preds, ones)

            fake = model['generator'](inp).detach()
            fake_preds = model["discriminator"](fake, inp)
            fake_targets = torch.zeros(fake.size(0), 1,fake_preds.shape[2], fake_preds.shape[3], device=device)
            fake_loss = criterion['discriminator'](fake_preds, fake_targets)

            loss_d = (real_loss + fake_loss)*0.5
            loss_d.backward()
            optimizer["discriminator"].step()
            loss_d_per_epoch.append(loss_d.item())

        for inp_val, _ in valloader:
            model["generator"].eval()
            with torch.no_grad():
                pic_val = model['generator'](inp_val)
 

        # Значения лоссов
        losses_g_train.append(np.mean(loss_g_per_epoch))
        losses_d_train.append(np.mean(loss_d_per_epoch))
     
        # Отрисовка результата
        clear_output(wait=True)
        plot_train(inp, out, fake, mode='Train')
        
        show_images(inp_val.cpu(), 6, figsize=(17, 2.5),title='Valllidate')
        show_images(pic_val.cpu(), 6, figsize=(17, 2.5))

        
        # Отображение графиков лоссов после каждой эпохи
        plt.figure(figsize=(16, 3))
        plt.plot(np.arange(epoch+1), losses_g_train, label='Train')
        plt.xlim(0, epochs)
        plt.ylabel('Generator Loss')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(16, 3))
        plt.plot(np.arange(epoch+1), losses_d_train, label='Train')
        plt.xlim(0, epochs)
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Discriminator Loss')
        plt.legend()
        plt.show()

    return losses_g_train, losses_d_train

train(pix2pix, trainloader_facade, valloader_facade)

# Сохранение модели

torch.save({
    'model_discriminator_state_dict': pix2pix['discriminator'].state_dict(),
    'model_generator_state_dict': pix2pix['generator'].state_dict()
}, path)
