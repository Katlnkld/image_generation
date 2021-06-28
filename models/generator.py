import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()       
        
        # Энкодер

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
      
        # Декодер
        
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5)
        )      
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512*2, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5)
        ) 
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512*2, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.5)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512*2, out_channels=512, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )    
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512*2, out_channels=256, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )   
        self.deconv5 = nn.Sequential(    
            nn.ConvTranspose2d(in_channels=256*2, out_channels=128, kernel_size=4, stride =2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deconv6 = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=128*2, out_channels=64, kernel_size=4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deconv7 = nn.Sequential( 
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride = 2, padding = 1)
        )

    def forward(self, x):
        e0 = self.conv0(x)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)

        d0 = self.deconv0(e7)
        d1 = self.deconv1(torch.cat([e6,d0], dim=1))
        d2 = self.deconv2(torch.cat([e5,d1], dim=1))
        d3 = self.deconv3(torch.cat([e4,d2], dim=1))
        d4 = self.deconv4(torch.cat([e3,d3], dim=1))
        d5 = self.deconv5(torch.cat([e2,d4], dim=1))
        d6 = self.deconv6(torch.cat([e1,d5], dim=1))
        d7 = self.deconv7(d6)
        
        out = torch.tanh(d7)
        
        return out