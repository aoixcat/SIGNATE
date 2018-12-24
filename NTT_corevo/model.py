import torch
import torch.nn as nn
import torch.nn.functional as F
        
class VAE(nn.Module):
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(VAE, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 8, (3,9), (1,1), padding=(1, 4))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(1, 8, (3,9), (1,1), padding=(1, 4))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16, 16, (4,8), (2,2), padding=(1, 3))
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv3_gated = nn.Conv2d(16, 16, (4,8), (2,2), padding=(1, 3))
        self.conv3_gated_bn = nn.BatchNorm2d(16)
        self.conv3_sigmoid = nn.Sigmoid()
        
        self.conv4_mu = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        self.conv4_logvar = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(5, 16, (9,5), (9,1), padding=(0, 2))
        self.upconv1_bn = nn.BatchNorm2d(16)
        self.upconv1_gated = nn.ConvTranspose2d(5, 16, (9,5), (9,1), padding=(0, 2))
        self.upconv1_gated_bn = nn.BatchNorm2d(16)
        self.upconv1_sigmoid = nn.Sigmoid()
        
        self.upconv2 = nn.ConvTranspose2d(16, 16, (4,8), (2,2), padding=(1, 3))
        self.upconv2_bn = nn.BatchNorm2d(16)
        self.upconv2_gated = nn.ConvTranspose2d(16, 16, (4,8), (2,2), padding=(1, 3))
        self.upconv2_gated_bn = nn.BatchNorm2d(16)
        self.upconv2_sigmoid = nn.Sigmoid()
        
        self.upconv3 = nn.ConvTranspose2d(16, 8, (4,8), (2,2), padding=(1, 3))
        self.upconv3_bn = nn.BatchNorm2d(8)
        self.upconv3_gated = nn.ConvTranspose2d(16, 8, (4,8), (2,2), padding=(1, 3))
        self.upconv3_gated_bn = nn.BatchNorm2d(8)
        self.upconv3_sigmoid = nn.Sigmoid()
        
        self.upconv4_mu = nn.ConvTranspose2d(8, 2//2, (3,9), (1,1), padding=(1, 4))
        self.upconv4_logvar = nn.ConvTranspose2d(8, 2//2, (3,9), (1,1), padding=(1, 4))

    def encode(self, x):
       
        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated))
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
        
        h4_mu = self.conv4_mu(h3)
        h4_logvar = self.conv4_logvar(h3)
       
        return h4_mu, h4_logvar 

    def decode(self, z):
        
        h5_ = self.upconv1_bn(self.upconv1(z))
        h5_gated = self.upconv1_gated_bn(self.upconv1(z))
        h5 = torch.mul(h5_, self.upconv1_sigmoid(h5_gated)) 
        
        h6_ = self.upconv2_bn(self.upconv2(h5))
        h6_gated = self.upconv2_gated_bn(self.upconv2(h5))
        h6 = torch.mul(h6_, self.upconv2_sigmoid(h6_gated)) 
        
        h7_ = self.upconv3_bn(self.upconv3(h6))
        h7_gated = self.upconv3_gated_bn(self.upconv3(h6))
        h7 = torch.mul(h7_, self.upconv3_sigmoid(h7_gated)) 
        
        h8_mu = self.upconv4_mu(h7)
        h8_logvar = self.upconv4_logvar(h7)
        
        return h8_mu, h8_logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu_enc, logvar_enc = self.encode(x)
        z_enc = self.reparameterize(mu_enc, logvar_enc)
        mu_dec, logvar_dec = self.decode(z_enc)
        z_dec = self.reparameterize(mu_dec, logvar_dec)
        return z_dec, mu_enc, logvar_enc
    
    # Reconstruction + KL divergence losses summed over all elements and batch
    def calc_loss(self, x):
        
        x = x.to(self.device)
        
        recon_x, mu, logvar = self.forward(x)
        
        L1 = torch.sum(torch.abs(recon_x - x))
        
        return L1
    
class Classifier(nn.Module):
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(VAE, self).__init__()
        
        self.conv1 = nn.Conv2d(5, 8, (4,1), (2,1), padding=(1, 0))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(1, 8, (4,1), (2,1), padding=(1, 1))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8, 16, (4,1), (2,1), padding=(1, 0))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8, 16, (4,1), (2,1), padding=(1, 0))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16, 32, (4,1), (2,1), padding=(1, 0))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_gated = nn.Conv2d(16, 32, (4,1), (2,1), padding=(1, 0))
        self.conv3_gated_bn = nn.BatchNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()
        
        self.conv4 = nn.Conv2d(32, 16, (4,1), (2,1), padding=(1, 0))
        self.conv4_bn = nn.BatchNorm2d(16)
        self.conv4_gated = nn.Conv2d(32, 16, (4,1), (2,1), padding=(1, 0))
        self.conv4_gated_bn = nn.BatchNorm2d(16)
        self.conv4_sigmoid = nn.Sigmoid()
        
        self.conv5 = nn.Conv2d(16, 2, (1,4), (1,2), padding=(0, 0))
        