import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.model_utils import calc_cd, MLP, calc_emd, knn_point, ChamferLoss, fps, cov
from utils.pointnet_utils import gather_points, group_points

class LFE(nn.Module):
    def __init__(self, mlp_c, mlp_f, npoints, k):
        super(LFE, self).__init__()
        self.mlp_cov = mlp_c
        self.mlp_f = mlp_f
        self.npoints = npoints
        self.k = k

    def forward(self, f, xyz):
        xyz_new = fps(xyz, self.npoints, BNC=True)  # (B, npoints, 3), (B, npoints)
        dist, idx = knn_point(xyz, xyz_new, self.k)
        f = group_points(f, idx)                    # (B, C, npoints, k)
        f = f.max(dim=-1)[0]                        # (B, C, npoints)

        _cov = cov(xyz, k=self.k, idx=idx)
        f_cov = self.mlp_cov(_cov)

        f = torch.cat([f, f_cov], dim=1)
        f = self.mlp_f(f)
        return f, xyz_new

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mlp = MLP(channels=[3, 32, 64])
        self.cov_sa1 = LFE(MLP(channels=[9, 32, 64]), MLP(channels=[64 + 64, 64, 128], last_act=False), npoints=512, k=8)
        self.cov_sa2 = LFE(MLP(channels=[9, 32, 64]), MLP(channels=[64 + 128, 128, 256], last_act=False), npoints=128, k=8)
        self.cov_sa3 = LFE(MLP(channels=[9, 32, 64]), MLP(channels=[64 + 256, 256, 1024], last_act=False), npoints=32, k=8)
        self.mlp_g = MLP(channels=[128+256+1024, 1024], conv_type='linear', last_act=False)

    def forward(self, x):
        f = self.mlp(x)
        xyz = x.transpose(2, 1).contiguous()             # (B, N, 3)

        f1, xyz_new = self.cov_sa1(f, xyz)                  # (B, 128, 512), (B, 512, 3)
        f2, xyz_new = self.cov_sa2(F.relu(f1), xyz_new)     # (B, 256, 128), (B, 128, 3)
        f3, xyz_new = self.cov_sa3(F.relu(f2), xyz_new)     # (B, 512, 32),  (B, 32, 3)

        g = f3.max(dim=-1)[0]
        f = torch.cat([f1.max(dim=-1)[0], f2.max(dim=-1)[0], g], dim=1) #(B, 128+256+512)
        g = g + self.mlp_g(f)
        return g

class PE(nn.Module):
    def __init__(self, k=16, r=4):
        super(PE, self).__init__()
        self.k = k
        self.r = r
        self.mlp1 = MLP(channels=[3, 32, 64], last_act=True)
        self.mlp2 = MLP(channels=[9, 32], last_act=True)
        self.mlp3 = MLP(channels=[64+32, 128, 3*self.r], last_act=False)
        self.attention = MLP(channels=[64, 128, 64], conv_type='2D', last_act=False)

    def forward(self, x):                                             # (B, 3, N)
        B, C, N = x.shape
        _cov, idx = cov(x.transpose(2, 1).contiguous(), k=self.k, index=True)

        f1 = self.mlp1(x)                                             # (B, 64, N)
        f_knn = group_points(f1, idx[:,:,1:]) - f1.unsqueeze(-1)      # (B, C, N, k)
        w = self.attention(f_knn).softmax(dim=-1)
        f1 = (w * f_knn).sum(dim=-1)                                  # (B, 64, N)

        f = torch.cat([f1, self.mlp2(_cov)], dim=1)                   # (B, 32+64, N)
        x = x.repeat(1,1,self.r) + 0.15 *self.mlp3(f).view(B, -1, self.r*N)
        return x

class Decoder(nn.Module):
    def __init__(self, num_coarse=1024, num_dense=2048):
        super(Decoder, self).__init__()
        self.num_coarse = num_coarse
        self.num_dense = num_dense
        self.r = int(self.num_dense//self.num_coarse)
        self.linear = MLP(channels=[1024, 1024, 3 * self.num_coarse], conv_type='linear',  last_act=False)
        self.up = PE(k=8, r =2)
        self.chamfer_loss = ChamferLoss()

    def forward(self, g, x):
        B, C = g.shape
        coarse = self.linear(g).view(B, 3, -1)           #(B, 3, 1024)

        dist1, dist2 = self.chamfer_loss(coarse.transpose(2, 1).contiguous(), x.transpose(2, 1).contiguous())
        idx = torch.topk(dist1, 512, largest=True)[1]
        hole = gather_points(coarse, idx.int(), tanspose=False)

        xx_merge = torch.cat([hole, x], dim=-1)          # (B, 3, 512+2048)
        xx = fps(xx_merge, self.num_coarse, BNC=False)   # (B, 3, 512)
        out = self.up(xx)                                # (B, 3, 2048)

        return coarse.transpose(2,1).contiguous(), out.transpose(2,1).contiguous()

class CovNet(pl.LightningModule):
    def __init__(self, num_coarse=1024, num_dense=2048, lrate=1e-4):
        self.save_hyperparameters()
        super(CovNet, self).__init__()
        self.lr = lrate
        self.num_coarse = num_coarse
        self.num_dense = num_dense
        self.E = Encoder()
        self.D = Decoder(num_coarse=num_coarse, num_dense=num_dense)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        y_coarse, y_detail = self.D(self.E(x), x)
        return y_coarse, y_detail

    def share_step(self, batch):
        x, fine = batch                                 # (B,N,3)
        y_coarse, y_detail = self.forward(x)            # (B,512,3), (B,2048,3)
        coarse = fps(fine, y_coarse.shape[1], BNC=True)

        emd2 = calc_emd(y_detail, fine)
        rec_loss =  calc_emd(y_coarse, coarse).mean() + emd2.mean()
        return rec_loss

    def training_step(self, batch, batch_idx):
        return self.share_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.share_step(batch)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.7)
        return [opt], [sched]


if __name__ == "__main__":
    x = torch.rand(2, 1024, 3).cuda()
    model = CovNet(num_coarse=1024, num_dense=16384, lrate=1e-4).cuda()
    coarse, fine = model(x)
    print(coarse.shape, fine.shape)

