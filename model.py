from torch import nn
import torch
from torch.nn import functional as F
from tqdm import tqdm


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
    
    def forward(self, inputs,labels):
        encoded_inputs = self.encoder(inputs)
        z, mus, logvars, zs = self.latent_operations(encoded_inputs, labels)
        return self.decoder(z), zs, mus, logvars


def loss_func(recon_x, x, mus, logvars):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    mse = mse_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
    return mse + kld, mse, kld


def train_epoch(model, train_dataloader, loss_func, optimizer, epoch):
    model.train()
    train_loss = 0
    train_mse_loss = 0
    train_kld_loss = 0
    n = len(train_dataloader.dataset)
    for i, batch in enumerate(tqdm(train_dataloader)):
        scene = batch['scene'].to('cuda')
        masks = batch['masks'].to('cuda')
        labels = batch['labels'].to('cuda')
        recon_scene, _, mus, logvars = model(masks, labels)
        loss, mse_loss, kld_loss = loss_func(recon_scene, scene, mus, logvars)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        train_mse_loss += mse_loss.item()
        train_kld_loss += kld_loss.item()
    print('====> Epoch: {} Train loss: {:.4f}, MSE loss: {:.4f}'.format(
          epoch, train_loss / len(train_dataloader.dataset), train_mse_loss / len(train_dataloader.dataset)))
    return train_loss / n, train_mse_loss / n, train_kld_loss / n


def test_epoch(model, test_dataloader, loss_func, epoch):
    model.eval()
    test_loss = 0
    test_mse_loss = 0
    test_kld_loss = 0
    n = len(test_dataloader.dataset)
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            scene = batch['scene'].to('cuda')
            masks = batch['masks'].to('cuda')
            labels = batch['labels'].to('cuda')
            recon_scene, _, mus, logvars = model(masks, labels)
            loss, mse_loss, kld_loss = loss_func(recon_scene, scene, mus, logvars)
            test_loss += loss.item()
            test_mse_loss += mse_loss.item()
            test_kld_loss += kld_loss.item()
        print('====> Epoch: {} Test loss: {:.4f}, MSE loss: {:.4f}'.format(
            epoch, test_loss / len(test_dataloader.dataset), test_mse_loss / len(test_dataloader.dataset)))
    return test_loss / n, test_mse_loss / n, test_kld_loss / n


def train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs=20, scheduler=None):
    min_loss = float('inf')
    train_losses = []
    test_losses = []
    mse_train = []
    mse_test = []
    kld_train = []
    kld_test = []
    for epoch in range(num_epochs):
        train_loss, mse_loss_train, kld_loss_train = train_epoch(model, train_dataloader, loss_func, optimizer, epoch)
        test_loss, mse_loss_test, kld_loss_test = test_epoch(model, test_dataloader, loss_func, epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        mse_train.append(mse_loss_train)
        mse_test.append(mse_loss_test)
        kld_train.append(kld_loss_train)
        kld_test.append(kld_loss_test)
        if scheduler is not None:
            scheduler.step()
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
    return train_losses, test_losses, mse_train, mse_test, kld_train, kld_test
