# model for VAE

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.dropout_prob = 0.2
        self.leak_slope = 0.2
        # 106x212x1 to latent vector
        self.dl1 = 16
        self.conv1 = nn.Conv2d(1, self.dl1, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(self.dl1)
        
        self.dl2 = 32
        self.conv2 = nn.Conv2d(self.dl1, self.dl2, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(self.dl2)

        self.dl3 = 48
        self.conv3 = nn.Conv2d(self.dl2, self.dl3, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(self.dl3)

        self.dl4 = 64
        self.conv4 = nn.Conv2d(self.dl3, self.dl4, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(4*11*self.dl4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_log_var = nn.Linear(512, z_dim)

    def forward(self, x):
        # Layer 1: input is 106x212x1, output is 52x106x16
        skip1 = self.conv1(x)
        x = self.bn1(skip1)
        x = F.relu(x, inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)

        # Layer 2: input is 52x106x16, output is 25x52x32
        skip2 = self.conv2(x)
        x = self.bn2(skip2)
        x = F.relu(x, inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)

        # Layer 3: input is 25x52x32, output is 11x25x64
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)

        # Layer 4: input is 11x25x64, output is 4x11x128
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = F.dropout2d(x, p=self.dropout_prob)
        
        #Layer 5: FC 6x6x128
        x = x.view(-1, 4*11*self.dl4)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, p=self.dropout_prob)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, p=self.dropout_prob)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var, skip1, skip2
    
class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()

        self.z_dim = z_dim #latent space dimension
        self.leak_slope = 0.01 #lrelu slope

        self.dl0 = 512 #layer depth
        self.fc = nn.Linear(z_dim, 6*13*self.dl0, bias=False)
        self.bn0 = nn.BatchNorm2d(self.dl0)
        
        self.dl1 = 256
        self.tconv1 = nn.ConvTranspose2d(self.dl0, self.dl1, kernel_size=4, stride=2, padding=1)#, bias=False)
        self.bn1 = nn.BatchNorm2d(self.dl1)

        self.dl2 = 128
        self.tconv2 = nn.ConvTranspose2d(self.dl1, self.dl2, kernel_size=4, stride=2, padding=1, output_padding=(1,0))#, bias=False)
        self.bn2 = nn.BatchNorm2d(self.dl2)

        self.dl3 = 64
        self.tconv3 = nn.ConvTranspose2d(self.dl2, self.dl3, kernel_size=(4,3), stride=2)#, bias=True)
        self.bn3 = nn.BatchNorm2d(self.dl3)

        self.tconv4 = nn.ConvTranspose2d(self.dl3, 1, kernel_size=4, stride=2)#, bias=True)
        self.sig = nn.Sigmoid()

    def forward(self, x, skip1=0, skip2=0):
        #Layer 0: FC 100-> 6x6x512
        x = self.fc(x)
        x = x.view(-1, self.dl0, 6, 13)
        x = self.bn0(x)
        x = F.relu(x,inplace=True)

        x = self.tconv1(x)
        x = self.bn1(x)
        x = F.relu(x,inplace=True)

        x = self.tconv2(x)
        x = self.bn2(x)
        x = F.relu(x,inplace=True)

        x = self.tconv3(x)
        x = self.bn3(x)
        #x = F.leaky_relu(x,negative_slope=self.leak_slope,inplace=True)
        x = self.sig(x)

        x = self.tconv4(x)
        x = self.sig(x)
        
        return x

class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, log_var, skip1, skip2 = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z, skip1, skip2)
        return x_hat, mu, log_var
    