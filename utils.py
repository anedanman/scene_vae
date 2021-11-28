import math
import torch
from tqdm import tqdm


def get_rot_matr(n, x1, x2, alpha):
    rad = torch.tensor(alpha * math.pi / 180)
    s = torch.sin(rad)
    c = torch.cos(rad)
    rot = torch.eye(n)
    rot[x1, x1] = c
    rot[x2, x2] = c
    rot[x1, x2] = -s
    rot[x2, x1] = s
    return rot


def loss_func(recon_x, x, mus, logvars):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    mse = mse_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp())
    return mse + kld, mse, kld


def train_epoch(model, train_dataloader, loss_func, optimizer, epoch, gamma=1):
    model.train()
    train_loss = 0
    train_mse_loss = 0
    train_kld_loss = 0
    train_discr_loss = 0
    n = len(train_dataloader.dataset)
    for i, batch in enumerate(tqdm(train_dataloader)):
        scene = batch['scene'].to('cuda')
        masks = batch['masks'].to('cuda')
        labels = batch['labels'].to('cuda')
        recon_scene, _, mus, logvars, discr_loss = model(masks, labels)
        loss, mse_loss, kld_loss = loss_func(recon_scene, scene, mus, logvars)
        loss += discr_loss * gamma
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        train_mse_loss += mse_loss.item()
        train_kld_loss += kld_loss.item()
        train_discr_loss += discr_loss.item()
    print('====> Epoch: {} Train loss: {:.4f}, MSE loss: {:.4f}, discr loss: {:.4f}'.format(
          epoch, train_loss / n, train_mse_loss / n, train_discr_loss / n))
    return train_loss / n, train_mse_loss / n, train_kld_loss / n, train_discr_loss / n


def test_epoch(model, test_dataloader, loss_func, epoch, gamma=1):
    model.eval()
    test_loss = 0
    test_mse_loss = 0
    test_kld_loss = 0
    test_discr_loss = 0
    n = len(test_dataloader.dataset)
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            scene = batch['scene'].to('cuda')
            masks = batch['masks'].to('cuda')
            labels = batch['labels'].to('cuda')
            recon_scene, _, mus, logvars, discr_loss = model(masks, labels)
            loss, mse_loss, kld_loss = loss_func(recon_scene, scene, mus, logvars)
            loss += discr_loss * gamma
            test_loss += loss.item()
            test_mse_loss += mse_loss.item()
            test_kld_loss += kld_loss.item()
            test_discr_loss += discr_loss.item()
        print('====> Epoch: {} Test loss: {:.4f}, MSE loss: {:.4f}, discr loss: {:.4f}'.format(
            epoch, test_loss / n, test_mse_loss / n, test_discr_loss / n))
    return test_loss / n, test_mse_loss / n, test_kld_loss / n, test_discr_loss / n
