# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/3 10:55
@Auth ： dongb
@File ：RDJSCC.py
@IDE ：PyCharm
"""
import torch.nn as nn
import torch
import torch.fft

from models import channel
from compressai.layers import GDN

from models.CVIE import CVIE

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def deconv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=0):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              output_padding=output_padding, bias=False)

class AF_block(nn.Module):
    def __init__(self, Nin, Nh, No, config):
        super(AF_block, self).__init__()
        self.fc1 = nn.Linear(Nin + 1, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.config = config

    def forward(self, x, snr):
        mu = torch.mean(x, (2, 3)).to(self.config["device"])
        snr = snr.to(self.config["device"])
        out = torch.cat((mu, snr), 1).to(self.config["device"])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out * x
        return out

class AF_block_Decoder(nn.Module):
    def __init__(self, Nin, Nh, No,config):
        super(AF_block_Decoder, self).__init__()
        self.fc1 = nn.Linear(Nin + 2, Nh)
        self.fc2 = nn.Linear(Nh, No)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.config = config

    def forward(self, x, snr1, snr2):
        mu = torch.mean(x, (2, 3)).to(self.config["device"])
        snr1 = snr1.to(self.config["device"])
        snr2 = snr2.to(self.config["device"])
        out = torch.cat((mu, snr1, snr2), 1).to(self.config["device"])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(2)
        out = out.unsqueeze(3)
        out = out * x
        return out

class Subnet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim_in,
                 kernel_size=5,
                 padding=2,
                 stride = 1,
                 norm_layer=nn.BatchNorm2d):
        super(Subnet, self).__init__()

        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        layers = [
            conv(in_channels, dim_in, kernel_size=kernel_size, padding=padding, stride=stride),
            norm_layer(dim_in),
            nn.ReLU(inplace=True),
            conv(dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride),
            norm_layer(dim_in),
            nn.ReLU(inplace=True),
            conv(dim_in, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        ]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class Encoder_stage(nn.Module):
    def __init__(self, set_type,
                 in_channels,
                 out_channels,
                 use_conv1x1=False,
                 kernel_size=5,
                 stride=1,
                 padding=1):
        super(Encoder_stage, self).__init__()
        self.conv1 = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = conv(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.use_conv1x1 = use_conv1x1
        if use_conv1x1 == True:
            self.conv3 = conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.gdn2(out)
        if self.use_conv1x1 == True:
            x = self.conv3(x)
        out = out + x
        out = self.prelu(out)
        return out

class Decoder_stage(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1,
                 output_padding=0):
        super(Decoder_stage, self).__init__()
        self.deconv1 = deconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              output_padding=output_padding)
        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                  output_padding=output_padding)

    def forward(self, x, activate_func='prelu'):
        out = self.deconv1(x)
        out = self.gdn1(out)
        out = self.prelu(out)
        out = self.deconv2(out)
        out = self.gdn2(out)
        if self.use_deconv1x1 == True:
            x = self.deconv3(x)
        out = out + x
        if activate_func == 'prelu':
            out = self.prelu(out)
        elif activate_func == 'sigmoid':
            out = self.sigmoid(out)
        return out

class CVIE_Decoder_stage(nn.Module):
    def __init__(self, in_channels, out_channels, use_deconv1x1=False, kernel_size=3, stride=1, padding=1,
                 output_padding=0, config=None):
        super(CVIE_Decoder_stage, self).__init__()
        self.deconv1 = CVIE(in_planes=in_channels, out_planes=out_channels, kernel_att = kernel_size, head = 2, kernel_conv = kernel_size, stride = 1, dilation = 1, config = config)

        self.deconv2 = deconv(out_channels, out_channels, kernel_size=1, stride=1, padding=0, output_padding=0)
        self.gdn1 = GDN(out_channels)
        self.gdn2 = GDN(out_channels)
        self.prelu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_deconv1x1 = use_deconv1x1
        if use_deconv1x1 == True:
            self.deconv3 = deconv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                  output_padding=output_padding)

    def forward(self, view1, view2, SNR1, SNR2, activate_func='prelu'):
        feat_pair_1, feat_pair_2  = self.deconv1(view1, view2, SNR1, SNR2), self.deconv1(view2, view1, SNR2, SNR1)
        feat_pair_1, sf1, sf2 = feat_pair_1[0],feat_pair_1[1],feat_pair_1[2]
        feat_pair_2 = feat_pair_2[0]
        feat_pair_1, feat_pair_2 = self.gdn1(feat_pair_1), self.gdn1(feat_pair_2)
        feat_pair_1, feat_pair_2 = self.prelu(feat_pair_1), self.prelu(feat_pair_2)
        feat_pair_1, feat_pair_2 = self.deconv2(feat_pair_1), self.deconv2(feat_pair_2)
        feat_pair_1, feat_pair_2 = self.gdn2(feat_pair_1), self.gdn2(feat_pair_2)
        if self.use_deconv1x1 == True:
            view1, view2 = self.deconv3(view1), self.deconv3(view2)
        feat_pair_1, feat_pair_2 = feat_pair_1 + view1, feat_pair_2 + view2
        if activate_func == 'prelu':
            feat_pair_1, feat_pair_2 = self.prelu(feat_pair_1), self.prelu(feat_pair_2)
        elif activate_func == 'sigmoid':
            feat_pair_1, feat_pair_2 = self.sigmoid(feat_pair_1), self.sigmoid(feat_pair_2)
        return feat_pair_1, feat_pair_2, sf1, sf2


class Encoder(nn.Module):
    def __init__(self, channel_N, kernel_sz, enc_N, padding_L, config):
        super(Encoder, self).__init__()
        Nh_AF = channel_N // 2
        self.encoder_stage1 = Encoder_stage(set_type="Conv", in_channels=3, out_channels=channel_N, use_conv1x1=True,
                                            kernel_size=kernel_sz, stride=2,
                                            padding=padding_L)
        self.encoder_stage2 = Encoder_stage(set_type="Conv", in_channels=channel_N, out_channels=channel_N,
                                            use_conv1x1=True, kernel_size=kernel_sz, stride=2,
                                            padding=padding_L)
        self.encoder_stage3 = Encoder_stage(set_type="Conv", in_channels=channel_N, out_channels=enc_N, use_conv1x1=True,
                                            kernel_size=kernel_sz, stride=1,
                                            padding=padding_L)
        self.AF1 = AF_block(channel_N, Nh_AF, channel_N, config)
        self.AF2 = AF_block(channel_N, Nh_AF, channel_N, config)
        self.AF3 = AF_block(enc_N, enc_N // 2, enc_N, config)

    def forward(self, z, SNR):
        z = self.encoder_stage1(z)
        z1 = self.AF1(z, SNR)
        z2 = self.encoder_stage2(z1)
        z2 = self.AF2(z2, SNR)
        z3 = self.encoder_stage3(z2)
        z3 = self.AF3(z3, SNR)
        return z1,z2,z3

class Decoder(nn.Module):
    def __init__(self, channel_N, kernel_sz, enc_N, padding_L,config,channel):
        super(Decoder, self).__init__()
        Nh_AF1 = enc_N // 2
        Nh_AF = channel_N // 2
        self.decoder_stage1 = CVIE_Decoder_stage(enc_N, channel_N, use_deconv1x1=True, kernel_size=kernel_sz, stride=1,
                                            padding=padding_L, config=config)
        self.decoder_stage2 = CVIE_Decoder_stage(channel_N, channel_N, use_deconv1x1=False, kernel_size=kernel_sz, stride=1,
                                            padding=padding_L, config=config)
        self.decoder_stage3 = CVIE_Decoder_stage(channel_N, channel_N, use_deconv1x1=True, kernel_size=kernel_sz,
                                            stride=1, padding=padding_L, config=config)
        self.decoder_stage4 = Decoder_stage(channel_N, channel_N, use_deconv1x1=True, kernel_size=kernel_sz, stride=2,
                                            padding=padding_L, output_padding=1)
        self.decoder_stage5 = Decoder_stage(channel_N, 3, use_deconv1x1=True, kernel_size=kernel_sz, stride=2,
                                            padding=padding_L, output_padding=1)
        self.AF1 = AF_block(enc_N, Nh_AF1, enc_N,config)
        self.AF2 = AF_block(channel_N, Nh_AF, channel_N,config)
        self.AF3 = AF_block(channel_N, Nh_AF, channel_N,config)
        self.AF4 = AF_block(channel_N, Nh_AF, channel_N,config)
        self.AF5 = AF_block(channel_N, Nh_AF, channel_N,config)
        self.config = config
        self.netCE = Subnet(in_channels=(config['N_pilot']*config['P']+1)*2, out_channels=config['P']*2, dim_in = 32)
        self.netEQ = Subnet(in_channels=(config['S']+1)*config['P']*2, out_channels=config['S']*config['P']*2, dim_in = 32)
        self.channel = channel

    def channel_estimation(self, out_pilot, noise_pwr):
        return channel.LMMSE_channel_est(self.channel.pilot,out_pilot, noise_pwr)

    def equalization(self, H_est, out_sig, noise_pwr):
        return channel.MMSE_equalization(H_est, out_sig, self.config['M'] * noise_pwr)

    def CE_EQ_process(self,out_pilot, noise_pwr, shape, out_sig):
        self.H_est_MMSE = self.channel_estimation(out_pilot, noise_pwr)
        sub11 = torch.view_as_real(self.channel.pilot).repeat(shape[0], 1, 1, 1, 1)
        sub12 = torch.view_as_real(out_pilot)
        sub1_input = torch.cat((sub11, sub12), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(shape[0],
                                                                                                               -1, shape[2],
                                                                                                               shape[3])
        sub1_output = self.netCE(sub1_input).view(shape[0], self.config['P'], 1, 2, self.config['M']).permute(0, 1, 2, 4, 3)
        self.H_est = self.H_est_MMSE + torch.view_as_complex(sub1_output.contiguous())
        self.rx = self.equalization(self.H_est, out_sig, noise_pwr)
        sub21 = torch.view_as_real(self.H_est)
        sub22 = torch.view_as_real(out_sig)
        sub2_input = torch.cat((sub21, sub22), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(shape[0],
                                                                                                               -1, shape[2],
                                                                                                               shape[3])
        sub2_output = self.netEQ(sub2_input).view(shape[0], self.config['P'], self.config['S'], 2, self.config['M']).permute(0, 1, 2, 4, 3)
        self.rx = self.rx + torch.view_as_complex(sub2_output.contiguous())
        dec_in = torch.view_as_real(self.rx).permute(0, 1, 2, 3, 4)
        return dec_in,self.H_est_MMSE

    def forward(self, view1, view2, SNR1, SNR2, out_pilot, out_sig, noise_pwr, out_pilot_y, out_sig_y, noise_pwr_y,
                shape, H_true, H_true_y):
        # # ----------------CSI estimation (model-driven)----------------------
        # view1_out, H_est_MMSE_view1 = self.CE_EQ_process(out_pilot, noise_pwr, shape, out_sig)
        # view2_out, H_est_MMSE_view2 = self.CE_EQ_process(out_pilot_y, noise_pwr_y, shape, out_sig_y)
        #
        # r1 = torch.view_as_real(self.channel.pilot).repeat(shape[0], 1, 1, 1, 1)
        # r2 = torch.view_as_real(out_pilot)
        # view1_out = torch.cat((r1, r2, view1_out), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(shape[0], -1, shape[2], shape[3])
        #
        # r1_y = torch.view_as_real(self.channel.pilot).repeat(shape[0], 1, 1, 1, 1)
        # r2_y = torch.view_as_real(out_pilot_y)
        # view2_out = torch.cat((r1_y, r2_y, view2_out), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(shape[0], -1, shape[2], shape[3])
        #
        # loss_H_view1 = torch.mean(torch.abs(H_est_MMSE_view1 - H_true) ** 2) * 2
        # loss_H_view2 = torch.mean(torch.abs(H_est_MMSE_view2 - H_true_y) ** 2) * 2
        # view1 = self.AF1(view1_out, SNR1)
        # view2 = self.AF1(view2_out, SNR2)
        # # ----------------------------------------------------

        # ----------------CSI estimation (data-driven)----------------------
        loss_H_view1, loss_H_view2 = 0, 0
        view1 = self.AF1(view1, SNR1)
        view2 = self.AF1(view2, SNR2)
        view1, view2, sf1, sf2 = self.decoder_stage1(view1, view2, SNR1, SNR2)

        view1_1 = self.AF2(view1, SNR1)
        view2_1 = self.AF2(view2, SNR2)
        view1_1, view2_1, sf3, sf4 = self.decoder_stage2(view1_1, view2_1, SNR1, SNR2)

        view1_2 = self.AF3(view1_1, SNR1)
        view2_2 = self.AF3(view2_1, SNR2)
        view1_2, view2_2, sf5, sf6 = self.decoder_stage3(view1_2, view2_2, SNR1, SNR2)

        view1_3 = self.AF4(view1_2, SNR1)
        view2_3 = self.AF4(view2_2, SNR2)
        view1_3 = self.decoder_stage4(view1_3)
        view2_3 = self.decoder_stage4(view2_3)

        view1_4 = self.AF5(view1_3, SNR1)
        view2_4 = self.AF5(view2_3, SNR2)
        view1_4 = self.decoder_stage5(view1_4, "sigmod")
        view2_4 = self.decoder_stage5(view2_4, "sigmod")

        return [view1, view1_1, view1_2, view1_3, view1_4], [view2, view2_1, view2_2, view2_3, view2_4], \
               [sf1, sf2, sf3, sf4, sf5, sf6], [loss_H_view1, loss_H_view2]


class Jscc(nn.Module):
    def __init__(self, config):
        super(Jscc, self).__init__()
        self.kernel_sz = 5
        self.padding_L = (self.kernel_sz - 1) // 2
        self.channel_N = 256
        self.config = config

        self.encoder = Encoder(self.channel_N, self.kernel_sz, config["enc_N"], self.padding_L, config)
        self.flatten = nn.Flatten()
        self.ofdm = channel.OFDM(config)
        self.decoder = Decoder(self.channel_N, self.kernel_sz, config["enc_N"]+6, self.padding_L, config,self.ofdm)

    def powerNorm(self, z, config):
        z_com = torch.view_as_complex(z.contiguous())
        pwr_z_com = torch.sum(torch.abs(z_com) ** 2, dim=-1) / torch.numel(z_com[0])
        z_com = z_com*config["power_factor"] / torch.sqrt(pwr_z_com.unsqueeze(-1))  # [64, 1024]
        return z_com

    def forward(self, view1, view2, SNR1, SNR2):
        # ===================Encoding View1
        view1_enc1, view1_enc2, view1_enc = self.encoder(view1, SNR1)
        view1_enc_v = view1_enc
        B, C, H, W = view1_enc.shape
        view1_enc = view1_enc.contiguous().view(B, self.config["P"], self.config["S"], 2, self.config["M"]) \
            .contiguous().permute(0, 1, 2, 4, 3)

        # ===================Power Normalize
        view1_enc = self.powerNorm(view1_enc, self.config)

        # ===================Pass channel
        out_pilot1, out_sig1, self.H_true, noise_pwr1, self.PAPR, self.PAPR_cp = self.ofdm(view1_enc, SNR=SNR1,
                                                                                           cof=None)
        self.H_true = self.H_true.to(self.config["device"])

        # ===================Decoding
        r1 = torch.view_as_real(self.ofdm.pilot).repeat(B, 1, 1, 1, 1)
        r2 = torch.view_as_real(out_pilot1)
        r3 = torch.view_as_real(out_sig1)
        dec_view1 = torch.cat((r1, r2, r3), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(B, -1, H, W)

        # ===================Encoding View2
        view2_enc1, view2_enc2, view2_enc = self.encoder(view2, SNR2)
        view2_enc_v = view2_enc
        B, C, H, W = view2_enc.shape
        shape = [B, C, H, W]
        view2_enc = view2_enc.contiguous().view(B, self.config["P"], self.config["S"], 2, self.config["M"]) \
            .contiguous().permute(0, 1, 2, 4, 3)

        # ===================Power Normalize
        view2_enc = self.powerNorm(view2_enc, self.config)

        # ===================Pass channel
        out_pilot2, out_sig2, self.H_true_y, noise_pwr2, self.PAPR_y, self.PAPR_cp_y = self.ofdm(view2_enc, SNR=SNR2,
                                                                                                 cof=None)
        self.H_true = self.H_true.to(self.config["device"])
        self.H_true_y = self.H_true_y.to(self.config["device"])

        # ===================Decoding
        r1_y = torch.view_as_real(self.ofdm.pilot).repeat(B, 1, 1, 1, 1)
        r2_y = torch.view_as_real(out_pilot2)
        r3_y = torch.view_as_real(out_sig2)
        dec_view2 = torch.cat((r1_y, r2_y, r3_y), 2).contiguous().permute(0, 1, 2, 4, 3).contiguous().view(B, -1, H, W)

        view1_hat, view2_hat, sf, loss_H = self.decoder(
            dec_view1, dec_view2, SNR1, SNR2,
            out_pilot1, out_sig1, noise_pwr1,
            out_pilot2, out_sig2, noise_pwr2,
            shape, self.H_true, self.H_true_y
        )

        r3 = r3.contiguous().view(B, C, H, W)
        r3_y = r3_y.contiguous().view(B, C, H, W)

        return view1_hat, view2_hat, view1_enc_v, view2_enc_v, \
               view1_enc1, view1_enc2, view2_enc1, view2_enc2, \
               r3, r3_y, sf, loss_H


if __name__ == "__main__":
    import yaml

    with open('../configs/config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    SNR_TRAIN1 = 2 * torch.ones((config["batchsize"], 1), dtype=torch.float32).to(config["device"])
    SNR_TRAIN2 = 3 * torch.ones((config["batchsize"], 1), dtype=torch.float32).to(config["device"])
    jscc = Jscc(config=config).to(config["device"])
    z = torch.randn(config["batchsize"], 3, 128, 256).to(config["device"])
    z_hat = jscc(z, z, SNR_TRAIN1, SNR_TRAIN1)
