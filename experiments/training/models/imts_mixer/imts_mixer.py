import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixerBlock(nn.Module):
    def __init__(
        self,
        channels,
        D,
    ):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            nn.RMSNorm([D, channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        self.kernel_mixer = nn.Sequential(
            nn.RMSNorm([channels, D]),
            nn.Linear(D, D),
            nn.ReLU(),
        )

    def forward(self, z):
        z = z + self.channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = z + self.kernel_mixer(z)
        return z


class Tuple_Encoder(nn.Module):

    def __init__(
        self,
        khd,
        D,
    ):
        super().__init__()

        self.nn = nn.Sequential(nn.Linear(2, khd), nn.ReLU(), nn.Linear(khd, D))

    def forward(self, X, T):
        return self.nn(torch.cat([X.unsqueeze(-1), T.unsqueeze(-1)], -1))


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        D,
        D_dec,
        kernel_hidden_dim,
        mixer_blocks=2,
    ):
        super().__init__()
        self.tuple_encoderH = Tuple_Encoder(
            khd=kernel_hidden_dim,
            D=D,
        )
        self.tuple_encoderW = Tuple_Encoder(
            khd=kernel_hidden_dim,
            D=D,
        )
        self.channel_bias = nn.Parameter(torch.randn(channels, D))
        self.channels = channels
        self.out_layer = nn.Sequential(
            nn.RMSNorm([channels, D]),
            nn.Linear(D, D_dec),
        )
        if mixer_blocks > 0:
            self.init_channel_mixer = nn.Sequential(
                nn.RMSNorm([D, channels]),
                nn.Linear(channels, channels),
                nn.ReLU(),
            )
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(channels=channels, D=D) for i in range(mixer_blocks - 1)]
        )
        self.use_init_mixer = mixer_blocks > 0

    def forward(self, T, X, M):
        # BS x T x C x KD
        H = self.tuple_encoderH(X=X, T=T)
        W = self.tuple_encoderW(X=X, T=T)
        W = W - (~M.unsqueeze(-1) * 10e9)
        W = W.softmax(dim=1)
        channel_mask = M.sum(1) > 0
        # z: BS x C x KD
        aggregated_channels = ((H * W).sum(1)) * channel_mask.unsqueeze(-1)
        z = torch.zeros_like(aggregated_channels)
        z += self.channel_bias.unsqueeze(0)
        z += aggregated_channels
        for m in self.mixer_blocks:
            z = z + m(z)
        # BS x C x KD
        if self.use_init_mixer:
            z = z + self.init_channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = self.out_layer(z)
        return z


class IMTSMixer(nn.Module):
    def __init__(
        self,
        channels,
        D,
        D_dec,
        kernel_hidden_dim,
        mixer_blocks,
        device,
    ) -> None:
        super().__init__()
        self.out_kernel = nn.Sequential(
            nn.Linear(1, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, D_dec * channels),
        )
        self.out = nn.Sequential(
            nn.Linear(D_dec, 1),
        )

        self.encoder = Encoder(
            channels=channels,
            D=D,
            D_dec=D_dec,
            kernel_hidden_dim=kernel_hidden_dim,
            mixer_blocks=mixer_blocks,
        )
        self.device = device

    def forward(self, T, X, M, YT, MY):
        # X_enc: BS x T xC x KD
        if T.dim() == 2:
            T = T.unsqueeze(-1).expand_as(X)
        z = self.encoder(T=T, X=X, M=M)
        # yhat: BS x YT x C x KD
        yhat = z.unsqueeze(1).expand(-1, YT.shape[1], -1, -1)
        query = self.out_kernel(YT.unsqueeze(-1)).unsqueeze(-1).reshape_as(yhat)

        return self.out(yhat * query).squeeze(-1)

    def forecasting_with_targets(self, batch_dict):
        X = batch_dict["observed_data"].to(self.device)
        T = batch_dict["observed_tp"].to(self.device)
        M = batch_dict["observed_mask"].to(torch.bool).to(self.device)
        Y = batch_dict["data_to_predict"].to(self.device)
        YT = batch_dict["tp_to_predict"].to(self.device)
        MY = batch_dict["mask_predicted_data"].to(self.device)
        YHAT = self.forward(T=T, X=X, M=M, YT=YT, MY=MY)
        return YHAT, Y, MY
