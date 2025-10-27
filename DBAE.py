import torch
import torch.nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn
import numpy as np

from pytorch_wavelets import DWTForward

def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''

    if not training:
        X_hat = (X - moving_min) / moving_mag
    else:
        assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat = (X - min_) / mag_
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat * gamma * alpha) + beta
    return Y, moving_mag.data, moving_min.data


class Stretch(nn.Module):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''

    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01 * torch.ones(shape))
        self.beta = nn.Parameter(np.pi * torch.ones(shape))
        self.register_buffer('moving_mag', 1. * torch.ones(shape))
        self.register_buffer('moving_min', np.pi * torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)
        Y, self.moving_mag, self.moving_min = stretch(
            X, self.alpha, self.gamma, self.beta, self.moving_mag, self.moving_min,
            eps=1e-5, momentum=0.99, training=self.training)
        return Y


class Equi_layer(nn.Module):
    def __init__(self, r2_act, hidden_in, hidden_out, h, k_size=3, p_size=1, equi=True):
        super(Equi_layer, self).__init__()
        self.hidden_in = hidden_in
        self.hidden_out = hidden_out

        self.in_type = enn.FieldType(r2_act, self.hidden_in * [r2_act.trivial_repr])
        self.out_type = enn.FieldType(r2_act, self.hidden_out * [r2_act.regular_repr])
        self.equi = equi
        self.h = h

        self.layer = enn.SequentialModule(
            enn.MaskModule(self.in_type, self.h, margin=1),
            enn.R2Conv(self.in_type, self.out_type, kernel_size=k_size, padding=p_size, bias=False),
            enn.InnerBatchNorm(self.out_type),
            enn.ReLU(self.out_type, inplace=True)
        )

    def forward(self, input):
        x = enn.GeometricTensor(input, self.in_type)
        x = self.layer(x)  # B, H * R, M, N
        if self.equi:
            x = x.tensor
            x = x.view(-1, self.hidden_out, 64, self.h - 10 + 3, self.h - 10 + 3)  # 64 means the number of rotations
            #print(x.shape)
            x = x.sum(2)
            #print(x.shape)
        return x

class feature_extractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(feature_extractor, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class feedback_encoder(nn.Module):
    def __init__(self, alpha, config, device):
        super(feedback_encoder, self).__init__()
        self.alpha = alpha
        self.config = config
        self.fc_hidden = config['fc_hidden']
        self.latent_dim = config['lat_dim']
        self.hidden_dims = config['hidden_dims']
        self.latent_dim = config['lat_dim']
        self.spool = nn.AdaptiveAvgPool2d(1)
        self.to_lat = nn.Linear(6, self.latent_dim)
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)
        self.block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(4, 4),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class encode_bloom(nn.Module):
    def __init__(self, alpha, config, device):
        super(encode_bloom, self).__init__()
        self.alpha = alpha
        self.config = config
        self.fc_hidden = config['fc_hidden']
        self.latent_dim = config['lat_dim']
        self.hidden_dims = config['hidden_dims']
        self.latent_dim = config['lat_dim']

        #self.encode_feat = feature_extractor(self.hidden_dims[-1] * 4, self.hidden_dims[-1] * 4)

        ''' self.bblock = nn.Sequential(
            nn.Conv2d(self.hidden_dims[-1] * 4, self.hidden_dims[-1] * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1] * 4),
            nn.ReLU()
        )'''

        self.spool = nn.AdaptiveAvgPool2d(1)
        self.to_lat = nn.Linear(self.hidden_dims[-1] * 4 * 2, self.latent_dim)
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)

        #self.adc = nn.Conv2d(8, self.hidden_dims[-1] * 4, kernel_size=1, stride=1)

    def forward(self, x_inv):
        #print(x_inv.shape)
        #[B, 128, 4, 4]

        x_inv = self.spool(x_inv)
        x_inv = torch.flatten(x_inv, start_dim=1)
        x_inv = self.to_lat(x_inv)
        # print(x_inv.shape)
        x_inv = self.strecth(x_inv)  # [B, lat_dim]

        return  x_inv

class deblur(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deblur, self).__init__()
        self.wt =DWTForward(J=1, mode='zero', wave='haar')
        self.conv_squeeze = nn.Conv2d(4, 4, 7, padding=3)

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.upscale_factor = 2
        self.upsample = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * (self.upscale_factor ** 2), kernel_size=3,
                      padding=1),
            nn.PixelShuffle(self.upscale_factor),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

        avg_w = torch.mean(x, dim=1, keepdim=True)
        max_w, _ = torch.max(x, dim=1, keepdim=True)
        agg = torch.cat([avg_w, max_w, avg_w, max_w], dim=1)

        sig = self.conv_squeeze(agg).sigmoid()
        #print(sig.size())
        x = torch.cat([yL * sig[:, 0, :, :].unsqueeze(1), y_HL * sig[:, 1, :, :].unsqueeze(1), y_LH * sig[:, 2, :, :].unsqueeze(1), y_HH * sig[:, 3, :, :].unsqueeze(1)], dim=1)
        x = self.conv_bn_relu(x)
        x = self.upsample(x)
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class fuseblock(nn.Module):
    def __init__(self, c, DW_Expand=1, dilations=[1, 1]):
        super(fuseblock, self).__init__()
        self.dw_channel = DW_Expand * c
        self.dilation1 = dilations[0]
        self.branch1 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=self.dilation1,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=self.dilation1)
        self.dilation2 = dilations[1]
        self.branch2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3,
                                 padding=self.dilation2,
                                 stride=1, groups=self.dw_channel,
                                 bias=True, dilation=self.dilation2)


        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c//2, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
    def forward(self, x1, x2):
        x_all = torch.cat([x1, x2], dim=1)

        z1 = self.branch1(x_all)
        z2 = self.branch2(x_all)
        z = z1 + z2
        z = self.sg1(z)   #c --> 0.5c
        x = self.sca(z) * z
        x = self.conv3(x)
        return x

class CODAE(nn.Module):
    def __init__(self, alpha, config, device):
        super(CODAE, self).__init__()
        self.config = config
        n_rot = 64
        self.r2_act = gspaces.rot2dOnR2(N=n_rot)
        self.hidden_dims = config['hidden_dims']   #1*8*16*32
        self.rhidden_dims = config['rhidden_dims']
        self.c, self.m, self.n = config['input_dim']  # self.input_dim
        self.reduced_dim_m = 4
        self.reduced_dim_n = 4
        self.fc_hidden = config['fc_hidden']
        self.latent_dim = config['lat_dim']
        self.alpha = alpha
        self.upscale_factor = 4
        self.device = device

        self.share0 = Equi_layer(self.r2_act, self.hidden_dims[0], self.hidden_dims[1], self.m, 10, 1, True)

        self.pool0 = nn.MaxPool2d(4, padding=2)

        self.share1 = Equi_layer(self.r2_act, self.hidden_dims[1], self.hidden_dims[2], 63, 10, 1, True)

        self.pool1 = nn.MaxPool2d(2, padding=1)

        self.share2 = Equi_layer(self.r2_act, self.hidden_dims[2], self.hidden_dims[3], 29, 10, 1, False)  #

        out_type = self.share2.out_type
        self.pool2 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=6))

        in_type = self.share2.out_type

        out_type = enn.FieldType(self.r2_act, self.hidden_dims[-1] * 4 * [self.r2_act.regular_repr])
        self.enc_inv1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
        self.encode_bloom = encode_bloom(alpha, config, device)
        self.spool = nn.AdaptiveAvgPool2d(1)
        self.gpool = enn.GroupPooling(out_type)

        self.to_lat = nn.Linear(self.hidden_dims[-1] * 4, self.latent_dim)
        self.strecth = Stretch(self.latent_dim, 2, self.alpha)

        self.enc_equi1 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[-1] * n_rot, self.hidden_dims[-1], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dims[-1]),
            nn.ReLU())

        self.et_fc = nn.Linear(self.hidden_dims[-1] * self.reduced_dim_m * self.reduced_dim_n, self.fc_hidden)
        self.to_et = nn.Linear(self.fc_hidden, 3)
        ##### decoding #####

        self.to_dec1 = nn.Linear(in_features=2*1, out_features=self.fc_hidden)
        self.de_fc1 = nn.Linear(self.fc_hidden, self.rhidden_dims[0] * self.reduced_dim_m * self.reduced_dim_n)

        self.to_dec2 = nn.Linear(in_features=2*1, out_features=self.fc_hidden)
        self.de_fc2 = nn.Linear(self.fc_hidden, self.rhidden_dims[0] * self.reduced_dim_m * self.reduced_dim_n)

        self.decoder_seq_s = nn.ModuleList()

        '''for idx in range(len(self.rhidden_dims) - 1):
            sub_seq_s = nn.ModuleList([nn.ConvTranspose2d(self.rhidden_dims[idx], self.rhidden_dims[idx + 1], 10, 4, padding=1),  # output_padding = 1),
                                     nn.BatchNorm2d(self.rhidden_dims[idx + 1]),
                                     nn.ReLU()])
            self.decoder_seq.append(sub_seq_s)'''

        '''self.decoder_seq_s.append(nn.ModuleList(
            [nn.ConvTranspose2d(self.rhidden_dims[0], self.rhidden_dims[1], 8, 4, padding=2),  # output_padding = 1),
             nn.BatchNorm2d(self.rhidden_dims[1]),
             nn.ReLU()]))
        self.decoder_seq_s.append(nn.ModuleList(
            [nn.ConvTranspose2d(self.rhidden_dims[1], self.rhidden_dims[2], 8, 4, padding=2),
             # output_padding = 1),
             nn.BatchNorm2d(self.rhidden_dims[2]),
             nn.ReLU()]))
        self.decoder_seq_s.append(nn.ModuleList(
            [nn.ConvTranspose2d(self.rhidden_dims[2], self.rhidden_dims[3], 8, 4, padding=2),
             # output_padding = 1),
             nn.BatchNorm2d(self.rhidden_dims[3]),
             nn.ReLU()]))'''
        self.decoder_seq_s.append(nn.ModuleList(
            [nn.Conv2d(self.rhidden_dims[0], self.rhidden_dims[1] * (self.upscale_factor ** 2), kernel_size=3,
                       padding=1),
             nn.PixelShuffle(self.upscale_factor),
             nn.ReLU(inplace=True)]
        ))
        self.db0 = deblur(self.rhidden_dims[1], self.rhidden_dims[1])

        self.decoder_seq_s.append(nn.ModuleList(
           [nn.Conv2d(self.rhidden_dims[1], self.rhidden_dims[2] * (self.upscale_factor ** 2), kernel_size=3,
                      padding=1),
            nn.PixelShuffle(self.upscale_factor),
            nn.ReLU(inplace=True)]
        ))

        self.db1 = deblur(self.rhidden_dims[2], self.rhidden_dims[2])

        self.decoder_seq_s.append(nn.ModuleList(
           [nn.Conv2d(self.rhidden_dims[2], self.rhidden_dims[3] * (self.upscale_factor ** 2), kernel_size=3,
                      padding=1),
            nn.PixelShuffle(self.upscale_factor),
            nn.ReLU(inplace=True)]
        ))

        self.db2 = deblur(self.rhidden_dims[3], self.rhidden_dims[3])

        self.decoder_seq_s.append(nn.ModuleList(
            [nn.Conv2d(self.rhidden_dims[3], self.rhidden_dims[3]//2, 3, padding=1),
             nn.BatchNorm2d(self.rhidden_dims[3]//2),
             nn.ReLU(inplace=True)
             ]
        ))

        self.db3 = deblur(self.rhidden_dims[3]//2, self.rhidden_dims[3]//2)

        self.decoder_seq_s.append(nn.ModuleList(
            [nn.Conv2d(self.rhidden_dims[3]//2, self.rhidden_dims[-1], kernel_size=3, padding=1),
             nn.Sigmoid()
             ]
        ))

        self.decoder_seq_b = nn.ModuleList()

        self.decoder_seq_b.append(nn.ModuleList(
            [nn.Conv2d(self.rhidden_dims[0], self.rhidden_dims[1] * (self.upscale_factor ** 2), kernel_size=3,
                       padding=1),
             nn.PixelShuffle(self.upscale_factor),
             nn.ReLU(inplace=True)]
        ))
        self.fuse_layer0 = fuseblock(self.rhidden_dims[1]*2, DW_Expand=1, dilations=[1, 1])

        self.decoder_seq_b.append(nn.ModuleList(
            [nn.Conv2d(self.rhidden_dims[1], self.rhidden_dims[2] * (self.upscale_factor ** 2), kernel_size=3,
                       padding=1),
             nn.PixelShuffle(self.upscale_factor),
             nn.ReLU(inplace=True)]
        ))

        self.fuse_layer1 = fuseblock(self.rhidden_dims[2] * 2, DW_Expand=1, dilations=[1, 2])

        self.decoder_seq_b.append(nn.ModuleList(
            [nn.Conv2d(self.rhidden_dims[2], self.rhidden_dims[3] * (self.upscale_factor ** 2), kernel_size=3,
                       padding=1),
             nn.PixelShuffle(self.upscale_factor),
             nn.ReLU(inplace=True)]
        ))

        self.fuse_layer2 = fuseblock(self.rhidden_dims[3] * 2, DW_Expand=1, dilations=[1, 3])

        self.decoder_seq_b.append(nn.ModuleList([nn.Conv2d(self.rhidden_dims[3], self.rhidden_dims[3]//2, 3, padding=1),
             nn.BatchNorm2d(self.rhidden_dims[3]//2),
             nn.ReLU(inplace=True)]))

        self.fuse_layer3 = fuseblock(self.rhidden_dims[3], DW_Expand=1, dilations=[2, 3])

        self.decoder_bhat = nn.Sequential(
            nn.Conv2d(self.rhidden_dims[3]//2, self.rhidden_dims[-1], kernel_size=3, padding=1),
            nn.Sigmoid()
        )


        self.pad = 10

        xgrid = np.linspace(-1, 1, self.m + self.pad * 2)
        ygrid = np.linspace(-1, 1, self.n + self.pad * 2)
        x0, x1 = np.meshgrid(xgrid, ygrid)
        grid = np.stack([x0.ravel(), x1.ravel()], 1)
        self.grid = torch.from_numpy(grid).float()

        self.feedback = feedback_encoder(alpha, config, device)

    def sample(self, num_samples=100, z=None):
        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)

        c = torch.cat((torch.cos(2 * np.pi * z), torch.sin(2 * np.pi * z)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T

        samples = self.decode_S(c)
        return samples

    def add_noise(self, z, epsilon=0.05):

        z[:,1] = z[:, 1] + ((torch.randn(z.shape[0], 1).to(self.device))*epsilon)[:,0]
        #print(z.shape)
        c = torch.cat((torch.cos(2 * np.pi * z), torch.sin(2 * np.pi * z)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T

        _, re_z_inv = self.decode_S(c)
        #re_z_inv = self.pool1(re_z_inv)
        return re_z_inv

    def encode_equi(self, x):

        x = self.share0(x)
        x = self.pool0(x)
        #print(x.shape)

        x = self.share1(x)
        x = self.pool1(x)
        #print(x.shape)
        x = self.share2(x)
        x = self.pool2(x)
        #print(x.shape)
        # x = [B, C',4, 4]
        x_inv = self.enc_inv1(x)
        x_inv = self.gpool(x_inv)
        x_inv = x_inv.tensor  ###to Tensor

        x_equi = x.tensor  ###to Tensor
        x_equi = self.enc_equi1(x_equi)
        #print(x_equi.shape)
        x_equi = torch.flatten(x_equi, start_dim=1)
        x_equi = self.et_fc(x_equi)
        x_equi = self.to_et(x_equi)

        return x_inv, x_equi



    def decode_S(self, x):
        x = nn.ReLU()(self.to_dec1(x))
        x = nn.ReLU()(self.de_fc1(x))
        #print(x.shape)

        x = x.view(-1, self.rhidden_dims[0], self.reduced_dim_m, self.reduced_dim_n)

        multi_s_hat = []
        for idx in range(len(self.decoder_seq_s)):
            for cidx in range(len(self.decoder_seq_s[idx])):
                x = self.decoder_seq_s[idx][cidx](x)
            if idx == 3:
                x = self.db3(x) + x
            elif idx == 2:
                x = self.db2(x) + x
            elif idx == 1:
                x = self.db1(x) + x
            elif idx == 0:
                x = self.db0(x) + x

            multi_s_hat.append(x)
        # x.shape = [16,1,1044,1044]
        #print(x.shape)


        return multi_s_hat

    def decode_B(self, x, s_hat):
        #len(s_hat) = 4
        x = nn.ReLU()(self.to_dec2(x))
        x = nn.ReLU()(self.de_fc2(x))
        # print(x.shape)

        x = x.view(-1, self.rhidden_dims[0], self.reduced_dim_m, self.reduced_dim_n)

        for idx in range(len(self.decoder_seq_b)):
            for cidx in range(len(self.decoder_seq_b[idx])):
                x = self.decoder_seq_b[idx][cidx](x)
            if idx == 3:
                x = x * s_hat[idx]
                #x = self.fuse_layer3(x, s_hat[idx]) + x
            elif idx == 2:
                #x = self.fuse_layer2(x, s_hat[idx]) + x
                x = x * s_hat[idx]
            elif idx == 1:
                #x = self.fuse_layer1(x, s_hat[idx]) + x
                x = x * s_hat[idx]
            elif idx == 0:
                #x = self.fuse_layer0(x, s_hat[idx]) + x
                x = x * s_hat[idx]
        b_hat = self.decoder_bhat(x)  #[1, 256, 256]

        # x.shape = [16,1,1044,1044]
        # print(x.shape)
        #b_hat = torch.cat([s_hat, x], dim=1)
        return b_hat

    def reparameterize(self, z):
        diff = torch.abs(z - z.unsqueeze(axis=1))
        none_zeros = torch.where(diff == 0., torch.tensor([100.]).to(z.device), diff)
        #print(none_zeros.shape)
        z_scores, _ = torch.min(none_zeros, axis=1)
        std = torch.normal(mean=0., std=1. * z_scores).to(z.device)
        s = z + std

        c = torch.cat((torch.cos(2 * np.pi * s), torch.sin(2 * np.pi * s)), 0)
        c = c.T.reshape(self.latent_dim * 2, -1).T

        return s,c

    def get_recon(self, x, z_equi):
        b = len(x)  #batch_size
        rot_mat = torch.cat((torch.cos(z_equi[:, 0]), -torch.sin(z_equi[:, 0]),
                             torch.sin(z_equi[:, 0]), torch.cos(z_equi[:, 0])),
                            axis=0).view(4, -1).T.view(-1, 2, 2)

        dxy = z_equi[:, 1:]
        recon_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "replicate")


        grid = self.grid.expand(b, recon_pad.size()[-2] * recon_pad.size()[-1], 2).to(z_equi.device)
        grid = grid - dxy.unsqueeze(1)  # translate coordinates
        grid = torch.bmm(grid, rot_mat)  # rotate coordinates by theta
        grid = grid.view(-1, recon_pad.size()[-2], recon_pad.size()[-1], 2)

        reconstruction = F.grid_sample(recon_pad, grid, padding_mode='border', align_corners=False, mode='bilinear')
        #reconstruction = F.grid_sample(recon_pad, grid)

        reconstruction = reconstruction[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return reconstruction

    def reconstr(self, input: torch.Tensor):
        z_inv, z_equi = self.encode_equi(input)
        z_inv = self.encode_bloom(z_inv)

        c_inv = torch.cat((torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0)
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T

        inv_recon,_ = self.decode_S(c_inv)

        reconstr = self.get_recon(inv_recon, z_equi)

        return reconstr, inv_recon

    def forward(self, input: torch.Tensor):
        #print(input.shape)
        z_inv_0, z_equi_0 = self.encode_equi(input)
        cor_input = self.get_recon(input, -z_equi_0)
        #z_inv_1 = self.feedback(cor_input)
        z_inv_1, z_equi_1 = self.encode_equi(cor_input)
        #print(z_inv.shape)
        z_equi = z_equi_0  + z_equi_1

        z_inv = self.encode_bloom(torch.cat([z_inv_0, z_inv_1], dim=1))
        #print(z_equi.shape)
        #print(z_inv.shape)
        s_inv, c_inv = self.reparameterize(z_inv)
        #print(c_inv.shape)
        c_b = c_inv[:, 0:2]
        c_s = c_inv[:, 2:4]
        inv_recon_shat = self.decode_S(c_s)
        inv_recon_bker = self.decode_B(c_b, inv_recon_shat)   #[1,256,256]
        inv_recon = inv_recon_bker   #[16,1,256,256]

        #inv_recon_back = self.feedback(inv_recon)  #[16,128,4,4]
        '''z_inv = self.encode_bloom(z_inv_0 + inv_recon_back)
        s_inv, c_inv = self.reparameterize(z_inv)
        c_b = c_inv[:, 0:2]
        c_s = c_inv[:, 2:4]
        inv_recon_shat = self.decode_S(c_s)
        inv_recon_bker = self.decode_B(c_b, inv_recon_shat)  # [1,256,256]
        inv_recon = inv_recon_bker
        '''

        recon = self.get_recon(inv_recon, z_equi)
        return recon, z_inv, z_equi, inv_recon_shat, inv_recon_bker

    def forward_eval(self, input: torch.Tensor, zinv1, zinv2):
        z_inv_0, z_equi_0 = self.encode_equi(input)
        cor_input = self.get_recon(input, -z_equi_0)
        # inv_recon_back = self.feedback(cor_input)
        z_inv_1, z_equi_1 = self.encode_equi(cor_input)
        # print(z_inv.shape)
        z_equi = z_equi_0 + z_equi_1

        z_inv = self.encode_bloom(torch.cat([z_inv_0, z_inv_1], dim=1))

        c_inv = torch.cat((torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0)
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T
        c_b = c_inv[:, 0:2]
        c_s = c_inv[:, 2:4]
        inv_recon_shat = self.decode_S(c_s)
        inv_recon = self.decode_B(c_b, inv_recon_shat)  # [1,256,256]
        recon = self.get_recon(inv_recon, z_equi)

        z_inv[:, 0] = float(zinv1)
        z_inv[:, 1] = float(zinv2)
        c_inv = torch.cat((torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0)
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T
        c_b = c_inv[:, 0:2]
        c_s = c_inv[:, 2:4]
        inv_recon_shat = self.decode_S(c_s)
        inv_recon = self.decode_B(c_b, inv_recon_shat)  # [1,256,256]
        recon_debloom = self.get_recon(inv_recon, z_equi)

        return recon, recon_debloom, inv_recon

    def forward_inv_new(self, input: torch.Tensor):
        z_inv, z_equi = self.encode_equi(input)
        z_inv = self.encode_bloom(z_inv)
        #print(z_inv)
        #print(z_equi)
        z_inv[:, 0] = 3.05
        #z_inv[:, 1] =3.15

        c_inv = torch.cat((torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0)
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T
        ''' c_s = c_inv[:, 0:2]
        c_b = c_inv[:, 2:4]'''

        inv_recon_shat = self.decode_S(c_inv )
        inv_recon_bker = self.decode_B(c_inv , inv_recon_shat)  # [1,256,256]
        inv_recon =  inv_recon_bker
        #inv_recon = self.outdecoder1(inv_recon)

        reconstr = self.get_recon(inv_recon, z_equi)

        #reconstr = self.outdecoder2(reconstr)

        return reconstr, z_equi

    def forward_lat(self, z):
        dxy, theta, z_inv = z[:, :2], z[:, 2], z[:, 3:]
        z_equi = torch.cat((theta.reshape(-1, 1), dxy), axis=-1)

        c_inv = torch.cat((torch.cos(2 * np.pi * z_inv), torch.sin(2 * np.pi * z_inv)), 0)
        c_inv = c_inv.T.reshape(self.latent_dim * 2, -1).T
        c_b = c_inv[:, 0:2]
        c_s = c_inv[:, 2:4]

        inv_recon_shat = self.decode_S(c_s)
        inv_recon_bker = self.decode_B(c_b, inv_recon_shat)   #[1,256,256]
        inv_recon =  inv_recon_bker
        #inv_recon = self.outdecoder1(inv_recon)

        reconstr = self.get_recon(inv_recon, z_equi)
        #reconstr = self.outdecoder2(reconstr)

        return reconstr
    def latent(self, input: torch.Tensor):
        z_inv_0, z_equi_0 = self.encode_equi(input)
        cor_input = self.get_recon(input, -z_equi_0)
        # inv_recon_back = self.feedback(cor_input)
        z_inv_1, z_equi_1 = self.encode_equi(cor_input)
        # print(z_inv.shape)
        z_equi = z_equi_0 + z_equi_1

        z_inv = self.encode_bloom(torch.cat([z_inv_0, z_inv_1], dim=1))

        theta = z_equi[:, 0]
        dxy = z_equi[:, 1:]
        return z_inv, theta, dxy