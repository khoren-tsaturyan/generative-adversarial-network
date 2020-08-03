import torch 
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self,coding_size):
        super(Generator,self).__init__()
        self.out_ch = 128
        self.fc1 = nn.Linear(coding_size,self.out_ch*16*4*4)
        self.tconv1 = nn.ConvTranspose2d(self.out_ch*16, self.out_ch*8,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_ch*8)
        self.tconv2 = nn.ConvTranspose2d(self.out_ch*8,self.out_ch*4,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(self.out_ch*4)
        self.tconv3 = nn.ConvTranspose2d(self.out_ch*4,self.out_ch*2,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_ch*2)
        self.tconv4 = nn.ConvTranspose2d(self.out_ch*2,self.out_ch,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(self.out_ch)
        self.tconv5 = nn.ConvTranspose2d(self.out_ch,3,kernel_size=4,stride=2,padding=1,bias=False)

    def forward(self,x):        
        x = self.fc1(x)
        x = x.view(-1,self.out_ch*16,4,4)
        x = self.bn1(F.selu(self.tconv1(x)))
        x = self.bn2(F.selu(self.tconv2(x)))
        x = self.bn3(F.selu(self.tconv3(x)))
        x = self.bn4(F.selu(self.tconv4(x)))
        x = torch.tanh(self.tconv5(x))

        return x 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        out_ch = 128
        self.conv1 = nn.Conv2d(3,out_ch,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch,out_ch*2,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch*2)
        self.conv3 = nn.Conv2d(out_ch*2,out_ch*4,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch*4)
        self.conv4 = nn.Conv2d(out_ch*4,out_ch*8,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch*8)
        self.conv5 = nn.Conv2d(out_ch*8,out_ch*16,kernel_size=4,stride=2,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(out_ch*16)
        self.conv6 = nn.Conv2d(out_ch*16,1,kernel_size=4,stride=1,padding=0,bias=False)
        self.out = nn.Linear(2048*16,1)

    def forward(self,x):
        x = self.bn1(F.leaky_relu(self.conv1(x)))
        x = F.dropout(x,0.4)
        x = self.bn2(F.leaky_relu(self.conv2(x)))
        x = F.dropout(x,0.4)
        x = self.bn3(F.leaky_relu(self.conv3(x)))
        x = F.dropout(x,0.4)
        x = self.bn4(F.leaky_relu(self.conv4(x)))
        x = F.dropout(x,0.4)
        x = self.bn5(F.leaky_relu(self.conv5(x)))
        x = F.dropout(x,0.4)
        x = x.view(-1,2048*16)
        x = torch.sigmoid(self.out(x))
        return x


