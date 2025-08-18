import os
import torch
import torch.nn as nn
import numpy as np
import yaml

def clipping(clipping_ratio, x):
    amp = x.abs()
    sigma = torch.sqrt(torch.mean(amp ** 2, -1, True))
    ratio = sigma * clipping_ratio / amp
    scale = torch.min(ratio, torch.ones_like(ratio))

    with torch.no_grad():
        bias = x * scale - x

    return x + bias


def add_cp(x, cp_len):
    return torch.cat((x[..., -cp_len:], x), dim=-1)

def rm_cp(x, cp_len):
    return x[..., cp_len:]

import torch.nn.functional as F
def batch_conv1d(x, weights):
    assert x.shape[0] == weights.shape[0]
    b, n = x.shape
    l = weights.shape[1]

    x = x.unsqueeze(0)  # 1xBxN
    weights = weights.unsqueeze(1)  # Bx1xL
    x = F.pad(x, (l - 1, 0), "constant", 0)  # 1xBx(N+L-1)
    out = F.conv1d(x, weight=weights, bias=None, stride=1, dilation=1, groups=b, padding=0)  # 1xBxN

    return out


def PAPR(x):
    power = torch.mean((x.abs()) ** 2, -1)
    pwr_max, _ = torch.max((x.abs()) ** 2, -1)
    return 10 * torch.log10(pwr_max / power)

def normalize(x, power):
    pwr = torch.mean(x.abs() ** 2, -1, True)
    return np.sqrt(power) * x / torch.sqrt(pwr)

def ZF_equalization(H_est, Y):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    return Y / H_est

def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xM
    # Y: NxPxSxM
    # no = complex_multiplication(Y, complex_conjugate(H_est))
    # de = complex_amp2(H_est)**2 + noise_pwr.unsqueeze(-1)
    # return no/de
    no = Y * H_est.conj()
    de = H_est.abs() ** 2 + noise_pwr.unsqueeze(-1)
    return no / de


def LS_channel_est(pilot_tx, pilot_rx):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM
    return torch.mean(pilot_rx, 2, True) / pilot_tx


def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM
    # return complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx))/(1+(noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]))
    return torch.mean(pilot_rx, 2, True) * pilot_tx.conj() / (1 + (noise_pwr.unsqueeze(-1) / pilot_rx.shape[2]))

# Realization of multipath channel as a nn module
class Channel(nn.Module):
    def __init__(self, config):
        super(Channel, self).__init__()
        # Generate unit power profile
        power = torch.exp(-torch.arange(config["L"]).float()/config["decay"]).view(1,1,config["L"])     # 1x1xL
        self.power = power/torch.sum(power)   # Normalize the path power to sum to 1
        self.config = config

    def sample(self, N, P, M, L):
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * (torch.randn(N, P, L) + 1j*torch.randn(N, P, L))
        cof_zp = torch.cat((cof, torch.zeros((N,P,M-L))), -1)
        H_t = torch.fft.fft(cof_zp, dim=-1)
        return cof, H_t

    def forward(self, input, cof=None):
        # Input size:   NxPx(Sx(M+K))
        # Output size:  NxPx(Sx(M+K))
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK = input.shape

        # If the channel is not given, random sample one from the channel model
        if cof is None:
            cof, H_t = self.sample(N, P, self.config["M"], self.config["L"])
        else:
            cof_zp = torch.cat((cof, torch.zeros((N,P,self.config["M"]-self.config["L"],2))), 2)
            cof_zp = torch.view_as_complex(cof_zp)
            H_t = torch.fft.fft(cof_zp, dim=-1)

        signal_real = input.real.float().view(N*P, -1)       # (NxP)x(Sx(M+K))
        signal_imag = input.imag.float().view(N*P, -1)       # (NxP)x(Sx(M+K))

        ind = torch.linspace(self.config["L"]-1, 0, self.config["L"]).long()
        cof_real = cof.real[...,ind].view(N*P, -1).float().to(self.config["device"]) # (NxP)xL
        cof_imag = cof.imag[...,ind].view(N*P, -1).float().to(self.config["device"]) # (NxP)xL

        output_real = batch_conv1d(signal_real, cof_real) - batch_conv1d(signal_imag, cof_imag)   # (NxP)x(L+SMK-1)
        output_imag = batch_conv1d(signal_real, cof_imag) + batch_conv1d(signal_imag, cof_real)   # (NxP)x(L+SMK-1)

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1).view(N,P,SMK,2)   # NxPxSMKx2
        output = torch.view_as_complex(output)

        return output, H_t

# Realization of OFDM system as a nn module
class OFDM(nn.Module):
    # def __init__(self, opt, device, pilot_path):
    def __init__(self, config):
        super(OFDM, self).__init__()
        # Setup the channel layer
        self.channel = Channel(config)
        self.config = config

        # Generate the pilot signal
        if not os.path.exists(config["pilot_path"]):
            bits = torch.randint(2, (config["M"],2))
            torch.save(bits,config["pilot_path"])
            pilot = (2*bits-1).float()
        else:
            bits = torch.load(config["pilot_path"])
            pilot = (2*bits-1).float()

        self.pilot = pilot.to(config["device"])
        self.pilot = torch.view_as_complex(self.pilot)
        self.pilot = normalize(self.pilot, 1)
        self.pilot_cp = add_cp(torch.fft.ifft(self.pilot), self.config["K"]).repeat(self.config["P"], self.config["N_pilot"],1)

    def forward(self, x, SNR, cof=None, batch_size=None):
        # Input size: NxPxSxM   The information to be transmitted
        # cof denotes given channel coefficients
        # If x is None, we only send the pilots through the channel
        is_pilot = (x == None)

        if not is_pilot:

            # Change to new complex representations
            N = x.shape[0]

            # IFFT:                    NxPxSxM  => NxPxSxM
            x = torch.fft.ifft(x, dim=-1)

            # Add Cyclic Prefix:       NxPxSxM  => NxPxSx(M+K)
            x = add_cp(x, self.config["K"])

            # Add pilot:               NxPxSx(M+K)  => NxPx(S+1)x(M+K)
            pilot = self.pilot_cp.repeat(N,1,1,1)
            x = torch.cat((pilot, x), 2)
            Ns = self.config["S"]
        else:
            N = batch_size
            x = self.pilot_cp.repeat(N,1,1,1)
            Ns = 0

        # Reshape:                 NxPx(S+1)x(M+K)  => NxPx(S+1)(M+K)
        x = x.view(N, self.config["P"], (Ns+self.config["N_pilot"])*(self.config["M"]+self.config["K"]))

        # PAPR before clipping
        papr = PAPR(x)

        # Clipping (Optional):     NxPx(S+1)(M+K)  => NxPx(S+1)(M+K)
        if self.config["is_clip"]:
            x = self.clip(x)

        # PAPR after clipping
        papr_cp = PAPR(x)

        # Pass through the Channel:        NxPx(S+1)(M+K)  =>  NxPx((S+1)(M+K))
        y, H_t = self.channel(x, cof)

        # Calculate the power of received signal
        pwr = torch.mean(y.abs()**2, -1, True) # NxPx((S+1)(M+K))  =>  NxPx1
        noise_pwr = pwr*10**(-SNR.view(N,1,1)/10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise

        # NxPx((S+S')(M+K))  =>  NxPx(S+S')x(M+K)
        output = y_noisy.view(N, self.config["P"], Ns+self.config["N_pilot"], self.config["M"]+self.config["K"])

        y_pilot = output[:,:,:self.config["N_pilot"],:]         # NxPxS'x(M+K)
        y_sig = output[:,:,self.config["N_pilot"]:,:]           # NxPxSx(M+K)

        if not is_pilot:
            # Remove Cyclic Prefix:
            info_pilot = rm_cp(y_pilot, self.config["K"])    # NxPxS'xM
            info_sig = rm_cp(y_sig, self.config["K"])        # NxPxSxM

            # FFT:
            info_pilot = torch.fft.fft(info_pilot, dim=-1)
            info_sig = torch.fft.fft(info_sig, dim=-1)

            return info_pilot, info_sig, H_t, noise_pwr, papr, papr_cp
        else:
            info_pilot = rm_cp(y_pilot, self.config["K"])    # NxPxS'xM
            info_pilot = torch.fft.fft(info_pilot, dim=-1)

            return info_pilot, H_t, noise_pwr

def complex_multiplication(x1, x2):
    real1 = x1[...,0]
    imag1 = x1[...,1]
    real2 = x2[...,0]
    imag2 = x2[...,1]

    out_real = real1*real2 - imag1*imag2
    out_imag = real1*imag2 + imag1*real2

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_conjugate(x):
    out_real = x[...,0]
    out_imag = -x[...,1]
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)


def MMSE_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xM
    # Y: NxPxSxM
    # no = complex_multiplication(Y, complex_conjugate(H_est))
    # de = complex_amp2(H_est)**2 + noise_pwr.unsqueeze(-1)
    # return no/de
    no = Y * H_est.conj()
    de = H_est.abs()**2 + noise_pwr.unsqueeze(-1)
    return no/de

def LMMSE_channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xM
    # pilot_rx: NxPxS'xM 32 1 2 64
    # return complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx))/(1+(noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]))
    return torch.mean(pilot_rx, 2, True) * pilot_tx.conj() / (
                1 + (noise_pwr.unsqueeze(-1) / pilot_rx.shape[2]))  # 32 1 1 64


def complex_amp(x):
    real = x[...,0]
    imag = x[...,1]
    return torch.sqrt(real**2 + imag**2).unsqueeze(-1)

# Realization of direct transmission over the multipath channel
class PLAIN(nn.Module):

    def __init__(self, config):
        super(PLAIN, self).__init__()
        self.config = config

        # Setup the channel layer
        self.channel = Channel(config)

    def forward(self, x, SNR):

        # Input size: NxPxM
        N, P, M = x.shape
        y,_ = self.channel(x, None)

        # Calculate the power of received signal
        pwr = torch.mean(y.abs()**2, -1, True)
        noise_pwr = pwr*10**(-SNR/10)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * (torch.randn_like(y) + 1j*torch.randn_like(y))
        y_noisy = y + noise                                    # NxPx(M+L-1)
        # rx = y_noisy[:, :, :M]
        rx = y_noisy
        return rx








