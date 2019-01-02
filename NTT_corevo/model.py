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
    
    def predict(self, x):
        shape = x.shape
        if  (len(shape) == 3):
            x = x.view(-1, shape[0], shape[1], shape[2])
        x.to(self.device)
        mu_enc, logvar_enc = self.encode(x)
        z_enc = self.reparameterize(mu_enc, logvar_enc)
        return z_enc
    
class Classifier(nn.Module):
    
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super(Classifier, self).__init__()
        
        self.fc1 = nn.Linear(5 * 1 * 256 // 2,  256 * 4)
        self.fc1_bn = nn.BatchNorm1d(256 * 4)
        self.fc1_gated = nn.Linear(5 * 1 * 256 // 2,  256 * 4)
        self.fc1_gated_bn = nn.BatchNorm1d(256 * 4)
        self.fc1_sigmoid = nn.Sigmoid()
        
        self.fc2 = nn.Linear(256 * 4,  256 * 4)
        self.fc2_bn = nn.BatchNorm1d(256 * 4)
        self.fc2_gated = nn.Linear(256 * 4,  256 * 4)
        self.fc2_gated_bn = nn.BatchNorm1d(256 * 4)
        self.fc2_sigmoid = nn.Sigmoid()
        
        self.fc3 = nn.Linear(256 * 4,  256 * 2)
        self.fc3_bn = nn.BatchNorm1d(256 * 2)
        self.fc3_gated = nn.Linear(256 * 4,  256 * 2)
        self.fc3_gated_bn = nn.BatchNorm1d(256 * 2)
        self.fc3_sigmoid = nn.Sigmoid()
        
        self.fc4 = nn.Linear(256 * 2,  256 * 1)
        self.fc4_bn = nn.BatchNorm1d(256 * 1)
        self.fc4_gated = nn.Linear(256 * 2, 256 * 1)
        self.fc4_gated_bn = nn.BatchNorm1d(256 * 1)
        self.fc4_sigmoid = nn.Sigmoid()
        
        self.fc5 = nn.Linear(256 * 1,  64)
        self.fc5_bn = nn.BatchNorm1d(64)
        self.fc5_gated = nn.Linear(256 * 1, 64)
        self.fc5_gated_bn = nn.BatchNorm1d(64)
        self.fc5_sigmoid = nn.Sigmoid()
        
        self.fc6 = nn.Linear(64, 1)
        self.fc6_sigmoid = nn.Sigmoid()
        
        """
        self.conv1 = nn.Conv2d(5, 8, (1,4), (1,2), padding=(0, 1))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(5, 8, (1,4), (1,2), padding=(0, 1))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8, 16, (1,4), (1,2), padding=(0, 1))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8, 16, (1,4), (1,2), padding=(0, 1))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16, 32, (1,4), (1,2), padding=(0, 1))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_gated = nn.Conv2d(16, 32, (1,4), (1,2), padding=(0, 1))
        self.conv3_gated_bn = nn.BatchNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()
        
        self.conv4 = nn.Conv2d(32, 16, (1,4), (1,2), padding=(0, 1))
        self.conv4_bn = nn.BatchNorm2d(16)
        self.conv4_gated = nn.Conv2d(32, 16, (1,4), (1,2), padding=(0, 1))
        self.conv4_gated_bn = nn.BatchNorm2d(16)
        self.conv4_sigmoid = nn.Sigmoid()
        
        self.conv5 = nn.Conv2d(16, 1, (1,1), (1,1), padding=(0, 0))
        self.conv5_sigmoid = nn.Sigmoid()
        """
        
        
    def classify(self, x):
        
        x = x.view(-1, 5 * 1 * 256 // 2)
        
        h1_ = self.fc1_bn(self.fc1(x))
        h1_gated = self.fc1_gated_bn(self.fc1_gated(x))
        h1 = torch.mul(h1_, self.fc1_sigmoid(h1_gated))
        
        h2_ = self.fc2_bn(self.fc2(h1))
        h2_gated = self.fc2_gated_bn(self.fc2_gated(h1))
        h2 = torch.mul(h2_, self.fc2_sigmoid(h2_gated))
        
        h3_ = self.fc3_bn(self.fc3(h2))
        h3_gated = self.fc3_gated_bn(self.fc3_gated(h2))
        h3 = torch.mul(h3_, self.fc3_sigmoid(h3_gated))
        
        h4_ = self.fc4_bn(self.fc4(h3))
        h4_gated = self.fc4_gated_bn(self.fc4_gated(h3))
        h4 = torch.mul(h4_, self.fc4_sigmoid(h4_gated))
        
        h5_ = self.fc5_bn(self.fc5(h4))
        h5_gated = self.fc5_gated_bn(self.fc5_gated(h4))
        h5 = torch.mul(h5_, self.fc5_sigmoid(h5_gated))
        
        h6 = self.fc6_sigmoid(self.fc6(h5))
        
        return h6
        
        """
        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated))
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated))
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated))
        
        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated))
        
        h5_ = F.softmax(self.conv5(h4), dim=1)
        h5 = torch.prod(h5_, dim=-1, keepdim=True)
        h5 = self.conv5_sigmoid(h5)
        
        return h5.view(-1, 1)
        """
    
    def calc_loss(self, x, self_label, label_):
        
        #Yes | No
        shape = label_.shape
        y_ = torch.zeros(shape[0], 1)
        for i in range(len(label_)):
            if (self_label == label_[i]):
                y_[i] = 1
        
        x = x.to(self.device)
        y_ = y_.to(self.device)
        
        y = self.classify(x)
        loss = F.binary_cross_entropy(y, y_)
        return loss
        
    def predict(self, x):
        x.to(self.device)
        y = self.classify(x)
        return y
        
class PredictingModel(nn.Module):
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(PredictingModel, self).__init__()
        
        # 1 * 32* 1024 -> 8 * 16 * 512
        self.conv1 = nn.Conv2d(1, 8, (4,8), (2,2), padding=(1, 3))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(1, 8, (4,8), (2,2), padding=(1, 3))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        # 8 * 16  * 512 -> 16 * 8 * 256 
        self.conv2 = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        # 16 * 8 * 256 -> 32 * 4 * 128 
        self.conv3 = nn.Conv2d(16, 32, (4,8), (2,2), padding=(1, 3))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv3_gated = nn.Conv2d(16, 32, (4,8), (2,2), padding=(1, 3))
        self.conv3_gated_bn = nn.BatchNorm2d(32)
        self.conv3_sigmoid = nn.Sigmoid()
        
        # 32 * 4 * 128 -> 32 * 4 * 64
        self.conv4 = nn.Conv2d(32, 32, (1,4), (1,2), padding=(0, 1))
        self.conv4_bn = nn.BatchNorm2d(32)
        self.conv4_gated = nn.Conv2d(32, 32, (1,4), (1,2), padding=(0, 1))
        self.conv4_gated_bn = nn.BatchNorm2d(32)
        self.conv4_sigmoid = nn.Sigmoid()
        
        # 32 * 4 * 64 -> 32 * 4 * 32
        self.conv5 = nn.Conv2d(32, 32, (1,4), (1,2), padding=(0, 1))
        self.conv5_bn = nn.BatchNorm2d(32)
        self.conv5_gated = nn.Conv2d(32, 32, (1,4), (1,2), padding=(0, 1))
        self.conv5_gated_bn = nn.BatchNorm2d(32)
        self.conv5_sigmoid = nn.Sigmoid()
        
        # 32 * 4 * 32 -> 16 * 4 * 16
        self.conv6 = nn.Conv2d(32, 16, (1,4), (1,2), padding=(0, 1))
        self.conv6_bn = nn.BatchNorm2d(16)
        self.conv6_gated = nn.Conv2d(32, 16, (1,4), (1,2), padding=(0, 1))
        self.conv6_gated_bn = nn.BatchNorm2d(16)
        self.conv6_sigmoid = nn.Sigmoid()
        
        # 16 * 4 * 16 -> 16 * 4 * 8
        self.conv7 = nn.Conv2d(16, 16, (1,2), (1,2), padding=(0, 0))
        self.conv7_bn = nn.BatchNorm2d(16)
        self.conv7_gated = nn.Conv2d(16, 16, (1,2), (1,2), padding=(0, 0))
        self.conv7_gated_bn = nn.BatchNorm2d(16)
        self.conv7_sigmoid = nn.Sigmoid()
        
        # 16 * 4 * 8 -> 32
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(16 * 4 * 8 // 2, 64)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(64, 6)
        
    def forward(self, x):
       
        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated))
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated))
        
        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated))
        
        h5_ = self.conv5_bn(self.conv5(h4))
        h5_gated = self.conv5_gated_bn(self.conv5_gated(h4))
        h5 = torch.mul(h5_, self.conv5_sigmoid(h5_gated)) 
        
        h6_ = self.conv6_bn(self.conv6(h5))
        h6_gated = self.conv6_gated_bn(self.conv6_gated(h5))
        h6 = torch.mul(h6_, self.conv6_sigmoid(h6_gated))
        
        h7_ = self.conv7_bn(self.conv7(h6))
        h7_gated = self.conv7_gated_bn(self.conv7_gated(h6))
        h7 = torch.mul(h7_, self.conv7_sigmoid(h7_gated))
        
        h8 = self.dropout1(h7)
        h8 = h8.view(-1, 16 * 4 * 8 // 2)
        h8 = F.relu(self.fc1(h8))
        h8 = self.dropout2(h8)
        h8 = self.fc2(h8)
        
        return F.softmax(h8, dim=1)
    
    def predict(self, x):
        shape = x.shape
        if  (len(shape) == 3):
            x = x.view(-1, shape[0], shape[1], shape[2])
        x.to(self.device)
        y = self.forward(x)
        return y
    
    def calc_loss(self, x, label):
        x = self.forward(x)
        loss = nn.CrossEntropyLoss()(x, label)
        return loss
    