import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dropout_prob = 0.4 #set dropout probability
        self.leak_slope = 0.2 #lrelu slope

        self.dl1 = 16 #layer depth
        self.conv1 = nn.Conv2d(1, self.dl1, kernel_size=4, stride=2)#, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dl1)

        self.dl2 = 32
        self.conv2 = nn.Conv2d(self.dl1, self.dl2, kernel_size=4, stride=2)#, padding=0, bias=False) removed padding and biases
        self.bn2 = nn.BatchNorm2d(self.dl2)
        
        self.dl3 = 48
        self.conv3 = nn.Conv2d(self.dl2, self.dl3, kernel_size=4, stride=2)#, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(self.dl3)

        self.dl4 = 64
        self.conv4 = nn.Conv2d(self.dl3, self.dl4, kernel_size=4, stride=2)#, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(self.dl4)

        self.fc = nn.Linear(4*11*self.dl4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Layer 1: input is 106x212x1, output is 52x106x16
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)

        # Layer 2: input is 52x106x16, output is 25x52x32
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)

        # Layer 3: input is 25x52x32, output is 11x25x64
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)

        # Layer 4: input is 11x25x64, output is 4x11x128
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)
        
        #Layer 5: FC 6x6x128
        x = x.view(-1, 4*11*self.dl4)
        x = self.fc(x)
        z = self.sigmoid(x)

        return z.squeeze()
    

    
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.z_dim = z_dim #latent space dimension
        self.leak_slope = 0.01 #lrelu slope

        self.dl0 = 512 #layer depth
        self.fc = nn.Linear(z_dim, 6*13*self.dl0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.dl0)
        
        self.dl1 = 256
        self.tconv1 = nn.ConvTranspose2d(self.dl0, self.dl1, kernel_size=4, stride=2, padding=1)#, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dl1)

        self.dl2 = 128
        self.tconv2 = nn.ConvTranspose2d(self.dl1, self.dl2, kernel_size=4, stride=2, padding=1)#, bias=False)
        self.bn2 = nn.BatchNorm2d(self.dl2)

        self.dl3 = 64
        self.tconv3 = nn.ConvTranspose2d(self.dl2, self.dl3, kernel_size=4, stride=2, padding=1)#, bias=True)
        self.bn3 = nn.BatchNorm2d(self.dl3)

        self.tconv4 = nn.ConvTranspose2d(self.dl3, 1, kernel_size=4, stride=2, padding=1)#, bias=True)
        self.tanh = nn.Sigmoid()

    def forward(self, z):
        #Layer 0: FC 100-> 6x6x512
        x = self.fc(z)
        x = x.view(-1, self.dl0, 6, 13)
        x = self.bn0(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)

        x = self.tconv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)

        x = self.tconv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)

        x = self.tconv3(x)
        #x = self.bn3(x)
        #x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)
        x = self.tanh(x)

        x = self.tconv4(x)
        x = self.tanh(x)

        return x
    