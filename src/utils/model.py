import torch.nn as nn
import torch
import fvdb.nn as fvnn

# TF32 in fVDB sparse conv requires Ampere+ (sm_80). Older GPUs must disable it.
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
    fvnn.SparseConv3d.allow_tf32 = False

def count_parameters(model, print_result=True):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_result:
        if num > 1e6:
            print("The model has {:.1f}M parameters".format(num/1000000))
        elif num > 1000:
            print("The model has {:.1f}k parameters".format(num/1000))
        return
    return num

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    emb = 2**emb * torch.pi
    emb = timesteps[:, None].float() * emb[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionCNN(nn.Module):
    def __init__(self, channels, layers=2, time_emb=6, one_layers=1, first_ks=3, in_channels=1, out_channels=1, dropout=.01):
        super(DiffusionCNN, self).__init__()
        self.out_channels = out_channels
        self.time_emb = time_emb
        self.net = [
            fvnn.SparseConv3d(in_channels+self.time_emb,
                              channels, kernel_size=first_ks, stride=1),
            fvnn.Dropout(dropout),
            fvnn.SiLU(inplace=True)]
        for _ in range(layers-1):
            self.net += [
                fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1),
                fvnn.Dropout(dropout),
                fvnn.SiLU(inplace=True)
            ]
        for _ in range(one_layers):
            self.net += [
                fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1),
                fvnn.SiLU(inplace=True)
            ]
        self.net.append(fvnn.SparseConv3d(
            channels, out_channels, kernel_size=1, stride=1))
        self.net = nn.Sequential(*self.net)

    def forward(self, x, t, cond=None):
        t = sinusoidal_embedding(t, self.time_emb)
        new_x = fvnn.VDBTensor(x.grid, x.grid.jagged_like(
            torch.cat((x.data.jdata, t), -1)))
        return self.net(new_x)


class _SparseLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a VDBTensor's jdata (N_voxels, C)."""
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: fvnn.VDBTensor) -> fvnn.VDBTensor:
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(self.ln(x.data.jdata)))


class DiffusionUNet(nn.Module):
    """
    U-Net variant of DiffusionCNN.

    All levels share the same channel width (`channels`).  Skip connections
    concatenate encoder and decoder features (2×channels) then project back
    to `channels` — no channel explosion at any resolution.

    LayerNorm after every conv keeps activations bounded across skip paths,
    preventing the gradient explosions that occur without normalization.

    Architecture per depth level:
      Encoder:  stride-2 down  →  LN  →  SiLU  →  ks=3  →  LN  →  Dropout  →  SiLU
      Decoder:  transposed up  →  cat(skip)  →  ks=1(2ch→ch)  →  LN  →  SiLU
                                             →  ks=3  →  LN  →  Dropout  →  SiLU
    Bottleneck: one_layers × (ks=3  →  LN  →  SiLU)
    """

    def __init__(self, channels, unet_depth=2, time_emb=6, one_layers=2,
                 first_ks=3, in_channels=1, out_channels=1, dropout=0.01):
        super().__init__()
        self.time_emb = time_emb
        self.unet_depth = unet_depth

        self.input_conv = nn.Sequential(
            fvnn.SparseConv3d(in_channels + time_emb, channels, kernel_size=first_ks, stride=1),
            _SparseLayerNorm(channels),
            fvnn.Dropout(dropout),
            fvnn.SiLU(inplace=True),
        )

        self.encoder_downs = nn.ModuleList()
        self.encoder_blocks = nn.ModuleList()
        for _ in range(unet_depth):
            self.encoder_downs.append(
                fvnn.SparseConv3d(channels, channels, kernel_size=2, stride=2)
            )
            self.encoder_blocks.append(nn.Sequential(
                _SparseLayerNorm(channels),
                fvnn.SiLU(inplace=True),
                fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1),
                _SparseLayerNorm(channels),
                fvnn.Dropout(dropout),
                fvnn.SiLU(inplace=True),
            ))

        bottleneck = []
        for _ in range(one_layers):
            bottleneck += [
                fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1),
                _SparseLayerNorm(channels),
                fvnn.SiLU(inplace=True),
            ]
        self.bottleneck = nn.Sequential(*bottleneck)

        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(unet_depth):
            self.decoder_ups.append(
                fvnn.SparseConv3d(channels, channels, kernel_size=2, stride=2, transposed=True)
            )
            self.decoder_blocks.append(nn.Sequential(
                fvnn.SparseConv3d(channels * 2, channels, kernel_size=1, stride=1),
                _SparseLayerNorm(channels),
                fvnn.SiLU(inplace=True),
                fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1),
                _SparseLayerNorm(channels),
                fvnn.Dropout(dropout),
                fvnn.SiLU(inplace=True),
            ))

        self.output_conv = fvnn.SparseConv3d(channels, out_channels, kernel_size=1, stride=1)

    def _cat(self, x: fvnn.VDBTensor, skip: fvnn.VDBTensor) -> fvnn.VDBTensor:
        merged = torch.cat([x.data.jdata, skip.data.jdata], dim=-1)
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(merged))

    def forward(self, x, t, cond=None):
        t = sinusoidal_embedding(t, self.time_emb)
        x = fvnn.VDBTensor(x.grid, x.grid.jagged_like(
            torch.cat((x.data.jdata, t), -1)))

        x = self.input_conv(x)

        skips = []
        for down, block in zip(self.encoder_downs, self.encoder_blocks):
            skips.append(x)
            x = down(x)
            x = block(x)

        x = self.bottleneck(x)

        for up, block, skip in zip(self.decoder_ups, self.decoder_blocks, reversed(skips)):
            x = up(x, out_grid=skip.grid)
            x = self._cat(x, skip)
            x = block(x)

        return self.output_conv(x)


class UpSampler(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, encoder_layers: int = 3, mult=2, dropout=.05):
        super().__init__()
        encoder = [fvnn.SparseConv3d(in_channels, hidden_channels, kernel_size=3, stride=1),
                   fvnn.Dropout(dropout),
                   fvnn.ReLU(inplace=True)]

        for _ in range(encoder_layers-1):
            encoder += [
                fvnn.SparseConv3d(
                    hidden_channels, hidden_channels, kernel_size=3, stride=1),
                fvnn.Dropout(dropout),
                fvnn.ReLU(inplace=True)
            ]
        self.encoder = nn.Sequential(*encoder)

        self.t_conv = fvnn.SparseConv3d(
            hidden_channels, hidden_channels, kernel_size=mult, stride=mult, transposed=True)

        self.decoder = nn.Sequential(*[
            fvnn.Dropout(dropout),
            fvnn.ReLU(inplace=True),
            fvnn.SparseConv3d(hidden_channels, hidden_channels,
                              kernel_size=1, stride=1),
            fvnn.Dropout(dropout),
            fvnn.ReLU(inplace=True),
            fvnn.SparseConv3d(hidden_channels, out_channels, kernel_size=1, stride=1)])

    def forward(self, input: fvnn.VDBTensor, x_upsample: fvnn.VDBTensor):
        x = self.encoder(input)
        x = self.t_conv(x, out_grid=x_upsample.grid)
        return self.decoder(x) + x_upsample
