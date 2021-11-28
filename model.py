from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from utils import get_rot_matr


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


class Autoencoder(nn.Module):
    def __init__(self, hidden_dims=None, in_channels=1):
        super(Autoencoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dims = [32, 64, 64, 64, 16] if hidden_dims is None else hidden_dims
        self.build_encoder()
        self.build_decoder()

    def build_encoder(self):
        modules = []
        in_channels = self.in_channels
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
    
    def build_decoder(self):
        modules = []
        in_channels = self.hidden_dims[-1]
        for h_dim in self.hidden_dims[-2::-1]:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       h_dim,
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[0],
                                    self.hidden_dims[0],
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    output_padding=1),
                nn.BatchNorm2d(self.hidden_dims[0]),
                nn.LeakyReLU(),
                nn.Conv2d(self.hidden_dims[0], out_channels=self.in_channels,
                          kernel_size= 3, padding= 1),
                nn.Tanh())
            )
        self.decoder = nn.Sequential(*modules)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
        self._mu1 = nn.ModuleList([nn.Linear(1024, embed_dim) for i in range(9)])
        self._sigma1 = nn.ModuleList([nn.Linear(1024, embed_dim) for i in range(9)])
        self._mu2 = nn.Linear(1024, embed_dim, bias=False)
        self._sigma2 = nn.Linear(1024, embed_dim, bias=False)
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
        self.final_act = nn.Tanh()
        self.latent_lins = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(9)])
        self.feed_fwd = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim))
        self.embed_dim = embed_dim
        self.discriminator = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1), nn.Sigmoid())
        self.discr_loss = nn.BCELoss(reduction='sum')
        
    def encode(self, x):
        x = self._conv_1(x)
        x = self._activation(x)      
        x = self._conv_2(x)
        x = self._activation(x)
        x = self._conv_3(x)
        x = self._residual_stack(x) 
        x = self._activation(x)
        x = self._cond1d(x) 
        x = self._activation(x)
        x = x.view(-1, 1024)
        return x
        
    def encoder(self, inputs):
        encoded_inputs = []
        for i in range(9):
            x = inputs[:, i, :, :, :]
            x = self.encode(x)
            encoded_inputs.append(x)
        return torch.stack(encoded_inputs)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def latent_operations(self, encoded_inputs, labels):
        mus = []
        logvars = []
        zs = []
        for i, inp in enumerate(encoded_inputs):
            mu1 = self._mu1[i](inp)
            sigma1 = self._sigma1[i](inp)
            z_i = self.reparameterize(mu1, sigma1)
            z_i = self.feed_fwd(z_i)
            z_i = self._activation(z_i)
            
            logvars.append(sigma1)
            mus.append(mu1)
            zs.append(z_i)

        zs = torch.stack(zs)
        mus = torch.stack(mus)
        logvars = torch.stack(logvars)

        zs = zs.transpose(0, 1)
        mask = labels.unsqueeze(-1).expand(zs.size())
        zs *= mask
        zs = zs.transpose(0, 1)
        z = torch.sum(zs, axis=0)

        mus = (mus.transpose(0, 1) * mask).transpose(0, 1)
        mus = torch.mean(mus, axis=0)

        logvars = (logvars.transpose(0, 1) * mask).transpose(0, 1)
        logvars = torch.mean(logvars, axis=0)
        return self._lin(z), mus, logvars, zs

    def discriminate(self, zs):
        rotated = []
        ys = []
        for zs_i in zs:
            xs = np.random.randint(1, self.embed_dim - 1, 2)
            x1, x2 = min(xs), max(xs)
            mode = np.random.randint(2)
            y = torch.ones(zs_i.shape[0]) if mode == 0 else torch.zeros(zs_i.shape[0])
            ys.append(y)
            alpha = np.random.randint(3, 13) if mode == 0 else np.random.randint(75, 105)
            rot_matr = get_rot_matr(self.embed_dim, x1, x2, alpha).to('cuda')
            zs_i_rot = zs_i @ rot_matr.t()
            rotated.append(zs_i_rot)
        zs = torch.flatten(zs, 0, 1)
        ys = torch.flatten(torch.stack(ys)).to('cuda').view(-1, 1)
        rotated = torch.flatten(torch.stack(rotated), 0, 1)
        res = torch.cat((zs, rotated), 1)
        pred = self.discriminator(res)
        loss = self.discr_loss(pred, ys)
        return loss

    def perceptual(self, zs, encoder_model):
        mse = torch.nn.MSELoss(reduction='sum')
        rotated_low = []
        rotated_high = []
        for zs_i in zs:
            xs = np.random.randint(1, self.embed_dim - 1, 2)
            x1, x2 = min(xs), max(xs)
            alpha = np.random.randint(3, 13)
            rot_matr = get_rot_matr(self.embed_dim, x1, x2, alpha).to('cuda')
            zs_i_rot = zs_i @ rot_matr.t()
            rotated_low.append(zs_i_rot)
        for zs_i in zs:
            xs = np.random.randint(1, self.embed_dim - 1, 2)
            x1, x2 = min(xs), max(xs)
            alpha = np.random.randint(75, 105)
            rot_matr = get_rot_matr(self.embed_dim, x1, x2, alpha).to('cuda')
            zs_i_rot = zs_i @ rot_matr.t()
            rotated_high.append(zs_i_rot)
        zs = self._lin(torch.flatten(zs, 0, 1))
        rotated_low = self._lin(torch.flatten(torch.stack(rotated_low), 0, 1))
        rotated_high = self._lin(torch.flatten(torch.stack(rotated_high), 0, 1))
        zs_perc = self.decoder(zs)
        zs1_perc = self.decoder(rotated_low)
        zs2_perc = self.decoder(rotated_high)
        with torch.no_grad():
            zs_perc = encoder_model(zs_perc)
            zs1_perc = encoder_model(zs1_perc)
            zs2_perc = encoder_model(zs2_perc)
        loss = mse(zs_perc, zs1_perc) - 0.5 * mse(zs_perc, zs2_perc)
        return loss

    def decoder(self, z):
        x = z.view(-1, 1, 32, 32)
        x = self._unpool(x)
        x = self._residual_stack2(x)
        x = self._activation(x)
        x = self._conv_trans_1(x)
        x = self._activation(x)
        x = self._conv_trans_2(x)
        x = self.final_act(x)
        return x
    
    def forward(self, inputs,labels, encoder_model):
        encoded_inputs = self.encoder(inputs)
        z, mus, logvars, zs = self.latent_operations(encoded_inputs, labels)
        perceptual_loss = self.perceptual(zs, encoder_model)
        #discr_loss = self.discriminate(zs)
        return self.decoder(z), zs, mus, logvars, perceptual_loss
