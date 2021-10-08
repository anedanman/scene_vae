from torch import nn
import torch
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
        #    nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
         #   nn.BatchNorm2d(num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
          #  nn.BatchNorm2d(num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class SceneVAE(nn.Module):
    
    def __init__(self, in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, embed_dim=1024):
        super(SceneVAE, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=4, 
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._activation = nn.GELU()
        self._avg_pool = nn.AvgPool2d(kernel_size=32)
        self._mu1 = nn.Linear(1024, embed_dim)
        self._sigma1 = nn.Linear(1024, embed_dim)
        self._mu2 = nn.Linear(1024, embed_dim)
        self._sigma2 = nn.Linear(1024, embed_dim)
        self._lin = nn.Linear(embed_dim, 1024)
        
        
        self._unpool = nn.ConvTranspose2d(in_channels=1, 
                                                out_channels=num_hiddens,
                                                kernel_size=3, 
                                                stride=1, padding=1)
        self._residual_stack2 = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=1,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self._cond1d = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=1,
                                 kernel_size=1,
                                 stride=1, padding=0)
        
    def encode(self, x):
        x = self._conv_1(x)
        x = self._activation(x)      
        x = self._conv_2(x)
        x = self._activation(x)
        x = self._conv_3(x)
        x = self._residual_stack(x) #(n_batch, 128, 32, 32)
        x = self._activation(x)
        x = self._cond1d(x) #(n_batch, 1, 32, 32)
        x = self._activation(x)
        x = x.view(-1, 1024)
        return x
        
    def encoder(self, inputs):
        #shape(inputs) = (n_batch, n_masks, C, W, H)
        encoded_inputs = []
        for i in range(9):
            x = inputs[:, i, :, :, :]
            x = self.encode(x)
            encoded_inputs.append(x)
        return encoded_inputs
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def latent_operations(self, encoded_inputs):
        z = None
        mus = []
        logvars = []
        zs = []
        for inp in encoded_inputs:
            mu1 = self._mu1(inp)
            mus.append(mu1)
            sigma1 = self._sigma1(inp)
            logvars.append(sigma1)
            z_i_1 = self.reparameterize(mu1, sigma1)
            z_i = z_i_1
            
            mu2 = self._mu2(inp)
            mus.append(mu2)
            sigma2 = self._sigma2(inp)
            logvars.append(sigma2)
            z_i_2 = self.reparameterize(mu2, sigma2)
            
            z_i = torch.fft.irfft(torch.fft.rfft(z_i_1, dim=1) * torch.fft.rfft(z_i_2, dim=1), dim=1)
            #z_i = z_i_1 * z_i_2
            if z is None:
                z = z_i
            else:
                z += z_i
            zs.append(z_i)
        return self._lin(z), mus, logvars, zs
    
    def decoder(self, z):
        x = z.view(-1, 1, 32, 32)
        x = self._unpool(x)
        x = self._residual_stack2(x)
        x = self._activation(x)
        x = self._conv_trans_1(x)
        x = self._activation(x)
        x = self._conv_trans_2(x)
        return x
    
    def forward(self, inputs):
        encoded_inputs = self.encoder(inputs)
        z, mus, logvars, zs = self.latent_operations(encoded_inputs)
        return self.decoder(z), zs, mus, logvars
