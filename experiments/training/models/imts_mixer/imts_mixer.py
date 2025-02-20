import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
    ):
        super().__init__()
        self.channel_mixer = nn.Sequential(
            nn.RMSNorm([hidden_dim, channels]),
            nn.Linear(channels, channels),
            nn.ReLU(),
        )
        self.kernel_mixer = nn.Sequential(
            nn.RMSNorm([channels, hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, z):
        # z: BS x C x D
        z = z + self.channel_mixer(z.transpose(1, 2)).transpose(1, 2)
        z = z + self.kernel_mixer(z)
        return z


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        hidden_dim,
        hidden_dim_dec,
        kernel_hidden_dim,
        mixer_blocks,
    ):
        super().__init__()
        kernel_output_dim = hidden_dim
        self.kernel = nn.Sequential(
            nn.Linear(1, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, kernel_output_dim),
        )

        self.init_kernel = nn.Sequential(
            nn.Linear(1, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, kernel_output_dim),
        )
        self.channel_bias = nn.Parameter(torch.randn(channels, hidden_dim))
        self.channels = channels
        self.out_layer = nn.Sequential(
            nn.RMSNorm([channels, hidden_dim]),
            nn.Linear(hidden_dim, hidden_dim_dec),
        )
        if mixer_blocks > 0:
            self.init_channel_mixer = nn.Sequential(
                nn.RMSNorm([hidden_dim, channels]),
                nn.Linear(channels, channels),
                nn.ReLU(),
            )
        self.mixer_blocks = nn.ModuleList(
            [
                MixerBlock(channels=channels, hidden_dim=hidden_dim)
                for i in range(mixer_blocks - 1)
            ]
        )
        self.use_init_mixer = mixer_blocks > 0

    def forward(self, T, X_enc, M):
        # BS x T x C x KD
        init_kernel = self.init_kernel(T.unsqueeze(-1))
        init_kernel = init_kernel.unsqueeze(-1).reshape_as(X_enc)
        Z_enc = X_enc * init_kernel
        # BS x T x C x KD
        weights = self.kernel(T.unsqueeze(-1))
        weights = weights.unsqueeze(-1).reshape_as(X_enc)
        weights = weights + X_enc
        weights = weights - (~M.unsqueeze(-1) * 10e9)
        weights = weights.softmax(dim=1)
        channel_mask = M.sum(1) > 0
        # z: BS x C x KD
        aggregated_channels = ((Z_enc * weights).sum(1)) * channel_mask.unsqueeze(-1)
        z = aggregated_channels
        z += self.channel_bias.unsqueeze(0)
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
        hidden_dim,
        hidden_dim_dec,
        kernel_hidden_dim,
        mixer_blocks,
        device,
    ) -> None:
        super().__init__()
        self.out_kernel = nn.Sequential(
            nn.Linear(1, kernel_hidden_dim),
            nn.ReLU(),
            nn.Linear(kernel_hidden_dim, hidden_dim_dec * channels),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim_dec, 1),
        )
        self.obs_encoder = nn.Linear(1, hidden_dim)
        self.encoder = Encoder(
            channels=channels,
            hidden_dim=hidden_dim,
            hidden_dim_dec=hidden_dim_dec,
            kernel_hidden_dim=kernel_hidden_dim,
            mixer_blocks=mixer_blocks,
        )
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, T, X, M, YT, MY):
        M = M.to(bool)
        # X_enc: BS x T xC x KD
        if T.dim() == 2:
            T = T.unsqueeze(-1).expand_as(X)
        X_enc = self.obs_encoder(X.unsqueeze(-1))
        X_enc = X.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        z = self.encoder(T=T, X_enc=X_enc, M=M)
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
