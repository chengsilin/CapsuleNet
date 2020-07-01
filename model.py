import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def squash(x):
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm_square = x_norm ** 2
    scale = x_norm_square / (1 + x_norm_square)
    x = scale * x / (x_norm + 1e-8)
    return x


def mask(x, y=None):
    x_norm = x.norm(dim=-1)
    if y is None:
        index = x_norm.max(dim=1)[1]
        mask = F.one_hot(index, num_classes=10).float()
    else:
        mask = y.float()
    masked = x * mask[:, :, None]
    masked = masked.view(masked.size()[0], -1)
    return x_norm, masked


def margin_loss(y_true, y_pred, x_recon, x, lam_recon):
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 \
        + 0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)
    return L_margin + lam_recon * L_recon


class PrimaryCapsule(nn.Module):
    def __init__(self, in_channels, out_channels, dim_capsule, kernel_size=9, strides=2, padding=0):
        super(PrimaryCapsule, self).__init__()
        self.dim_capsule = dim_capsule
        self.conv = nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=strides, padding=padding)

    def forward(self, x):
        batch_size, C, _, _ = x.size()
        out = self.conv(x)
        out = out.permute(0, 2, 3, 1)
        out = out.contiguous().view(batch_size, -1, self.dim_capsule)
        out = squash(out)
        return out


class CapsuleLayer(nn.Module):
    def __init__(self, out_num, in_num, out_dim, in_dim, routings=3):
        super(CapsuleLayer, self).__init__()
        self.in_dim = in_dim
        self.in_num = in_num
        self.out_dim = out_dim
        self.out_num = out_num
        self.routings = routings
        self.weight = nn.Parameter(torch.FloatTensor(out_num, in_num, out_dim, in_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=1.141)

    def forward(self, x):
        # x:(batch_size, in_num, in_dim)
        batch_size, in_num, in_dim = x.size()
        x = x.view(batch_size, 1, in_num, in_dim, 1)
        x_hat = torch.matmul(self.weight, x)
        # x_hat:(batch_size, out_num, in_num, out_dim)
        x_hat = x_hat.squeeze()
        x_hat_detach = x_hat.detach()

        b = torch.zeros([batch_size, self.out_num, self.in_num]).cuda()
        for i in range(self.routings):
            c = torch.softmax(b, dim=1)
            if i == (self.routings - 1):
                out = torch.sum(c[:, :, :, None] * x_hat, dim=-2)
                # out: (batch_size, out_num, 1, out_num)
                out = squash(out)
            else:
                out = torch.sum(c[:, :, :, None] * x_hat_detach, dim=-2)
                out = squash(out)
                b = b + torch.sum(torch.matmul(x_hat_detach, out[:, :, :, None]), dim=-1)
        out = out.squeeze(dim=-2)
        return out


class CapsuleNet(nn.Module):
    def __init__(self, args, in_num=6 * 6 * 32, in_dim=8, out_dim=16, n_class=10):
        super(CapsuleNet, self).__init__()
        self.input_size = args.input_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.relu = nn.ReLU()
        self.primaryCapsule = PrimaryCapsule(in_channels=256, out_channels=256, dim_capsule=in_dim)
        self.CapsuleLayer = CapsuleLayer(out_num=n_class, in_num=in_num, out_dim=out_dim, in_dim=in_dim,
                                         routings=args.routing_num)
        self.decoder = nn.Sequential(
            nn.Linear(n_class * out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, np.prod(self.input_size)),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        out = self.relu(self.conv1(x))
        out = self.primaryCapsule(out)
        out = self.CapsuleLayer(out)
        if self.training:
            pred, out = mask(out, y)
        else:
            pred, out = mask(out)
        out_reconstruct = self.decoder(out)
        out_reconstruct = out_reconstruct.reshape(-1, *self.input_size)
        return pred, out_reconstruct
