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


_COORD_DIMS = {'none': 0, 'z': 1, 'xyz': 3}


def grid_coord_features(grid, jidx, mode, h_ref, xy_ref):
    """Per-voxel absolute-position features for the model input.

    SparseConv3d is translation-equivariant, so the network has no notion of
    *where* a voxel sits — it cannot tell ground from sky.  This injects that
    notion explicitly as input channels, the standard CoordConv fix.

    The feature is height-above-the-sample's-lowest-voxel (≈ height above
    ground), normalized by a fixed physical reference `h_ref` (metres), so it is
    terrain-invariant and identical in meaning at train and inference time.
    'xyz' additionally adds per-sample-min-anchored x,y (normalized by xy_ref);
    'z' (recommended) keeps horizontal translation invariance, which is
    desirable for aerial scenes.  Returns (N, C) aligned to `grid.ijk` / jdata
    order, or None for mode='none'.
    """
    if mode == 'none':
        return None
    w = grid.grid_to_world(grid.ijk.float()).jdata   # (N, 3) world metres
    jidx = jidx.long()
    B = grid.grid_count

    def anchored(col, ref):
        v = w[:, col]
        vmin = torch.full((B,), float('inf'), device=w.device, dtype=w.dtype)
        vmin = vmin.scatter_reduce(0, jidx, v, reduce='amin', include_self=True)
        vmin = torch.where(torch.isinf(vmin), torch.zeros_like(vmin), vmin)
        return ((v - vmin[jidx]) / ref)[:, None]

    feats = [anchored(2, h_ref)]                      # z (height above ground)
    if mode == 'xyz':
        feats += [anchored(0, xy_ref), anchored(1, xy_ref)]
    return torch.cat(feats, dim=-1)


class DiffusionCNN(nn.Module):
    def __init__(self, channels, layers=2, time_emb=6, one_layers=1, first_ks=3, in_channels=1, out_channels=1, dropout=.01,
                 coord_features='none', coord_h_ref=30.0, coord_xy_ref=51.0):
        super(DiffusionCNN, self).__init__()
        self.out_channels = out_channels
        self.time_emb = time_emb
        self.coord_features = coord_features
        self.coord_h_ref = coord_h_ref
        self.coord_xy_ref = coord_xy_ref
        n_coord = _COORD_DIMS[coord_features]
        self.net = [
            fvnn.SparseConv3d(in_channels+self.time_emb+n_coord,
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
        parts = [x.data.jdata, t]
        coords = grid_coord_features(
            x.grid, x.data.jidx, getattr(self, 'coord_features', 'none'),
            getattr(self, 'coord_h_ref', 30.0), getattr(self, 'coord_xy_ref', 51.0))
        if coords is not None:
            parts.insert(1, coords.to(x.data.jdata.dtype))
        new_x = fvnn.VDBTensor(x.grid, x.grid.jagged_like(
            torch.cat(parts, -1)))
        return self.net(new_x)


class _SparseLayerNorm(nn.Module):
    """LayerNorm over the channel dim of a VDBTensor's jdata (N_voxels, C)."""
    def __init__(self, channels):
        super().__init__()
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: fvnn.VDBTensor) -> fvnn.VDBTensor:
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(self.ln(x.data.jdata)))


class GaussianFourierProjection(nn.Module):
    """
    Random-Fourier time embedding for continuous t ∈ [0,1] (EDM / NCSN++ style).
    """
    def __init__(self, embed_dim, scale=16.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.register_buffer("freqs", torch.randn(embed_dim // 2) * scale)

    def forward(self, t):  # t: (B,) in [0,1]  →  (B, embed_dim)
        proj = t[:, None] * self.freqs[None, :] * 2 * torch.pi
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation of a VDBTensor by a per-sample embedding.

    Produces per-channel (scale, shift) from the time embedding and broadcasts
    them to every voxel via the grid's per-voxel batch index (``jidx``), which
    fVDB keeps correct at every resolution — so σ reaches the bottleneck and
    decoder, not just the stem.
    """
    def __init__(self, temb_dim, channels):
        super().__init__()
        self.lin = nn.Linear(temb_dim, 2 * channels)
        # zero-init → blocks start as identity-modulated (stable warm start)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: fvnn.VDBTensor, temb) -> fvnn.VDBTensor:
        scale, shift = self.lin(temb).chunk(2, dim=-1)   # (B, C) each
        jidx = x.data.jidx.long()
        mod = x.data.jdata * (1 + scale[jidx]) + shift[jidx]
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(mod))


class _ConvLNStack(nn.Module):
    """A conv→LN sub-stack that defers activation, so FiLM can be applied
    between normalization and the nonlinearity (standard FiLM placement)."""
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FiLMEncoderBlock(nn.Module):
    """ks=3 → LN → FiLM → SiLU → ks=3 → LN → Dropout → SiLU (FiLM injects σ once)."""
    def __init__(self, channels, temb_dim, dropout):
        super().__init__()
        self.norm1 = _SparseLayerNorm(channels)
        self.film = FiLM(temb_dim, channels)
        self.act1 = fvnn.SiLU(inplace=True)
        self.conv2 = fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1)
        self.tail = nn.Sequential(
            _SparseLayerNorm(channels), fvnn.Dropout(dropout), fvnn.SiLU(inplace=True),
        )

    def forward(self, x, temb):
        x = self.norm1(x)
        x = self.film(x, temb)
        x = self.act1(x)
        x = self.conv2(x)
        return self.tail(x)


class FiLMDecoderBlock(nn.Module):
    """ks=1 (2ch→ch) → LN → FiLM → SiLU → ks=3 → LN → Dropout → SiLU."""
    def __init__(self, channels, temb_dim, dropout):
        super().__init__()
        self.proj = fvnn.SparseConv3d(channels * 2, channels, kernel_size=1, stride=1)
        self.norm1 = _SparseLayerNorm(channels)
        self.film = FiLM(temb_dim, channels)
        self.act1 = fvnn.SiLU(inplace=True)
        self.conv2 = fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1)
        self.tail = nn.Sequential(
            _SparseLayerNorm(channels), fvnn.Dropout(dropout), fvnn.SiLU(inplace=True),
        )

    def forward(self, x, temb):
        x = self.proj(x)
        x = self.norm1(x)
        x = self.film(x, temb)
        x = self.act1(x)
        x = self.conv2(x)
        return self.tail(x)


class FiLMBottleneckBlock(nn.Module):
    """ks=3 → LN → FiLM → SiLU, repeated `n` times — keeps σ present at the
    coarsest scale where one voxel covers a tile-wide physical area."""
    def __init__(self, channels, temb_dim, n):
        super().__init__()
        self.convs = nn.ModuleList(
            fvnn.SparseConv3d(channels, channels, kernel_size=3, stride=1) for _ in range(n))
        self.norms = nn.ModuleList(_SparseLayerNorm(channels) for _ in range(n))
        self.films = nn.ModuleList(FiLM(temb_dim, channels) for _ in range(n))
        self.act = fvnn.SiLU(inplace=True)

    def forward(self, x, temb):
        for conv, norm, film in zip(self.convs, self.norms, self.films):
            x = conv(x)
            x = norm(x)
            x = film(x, temb)
            x = self.act(x)
        return x


class DiffusionUNet(nn.Module):
    """
    U-Net variant of DiffusionCNN with FiLM noise conditioning.

    All levels share the same channel width (`channels`).  Skip connections
    concatenate encoder and decoder features (2×channels) then project back
    to `channels` — no channel explosion at any resolution.

    Noise level σ(t) is no longer concatenated to the input once at the stem
    (where it is averaged away by the first downsample).  Instead t is mapped
    through a Gaussian-Fourier embedding → 2-layer MLP → a shared time vector,
    and injected as FiLM (per-channel scale+shift) into *every* encoder,
    bottleneck and decoder block.  This is the standard ADM/EDM conditioning
    and is what lets a population denoiser resolve adjacent noise levels.

    `time_emb` is the Gaussian-Fourier dimension (≥128 recommended); the MLP
    width is `channels * 4`.

    Architecture per depth level:
      Encoder:  stride-2 down  →  [LN → FiLM → SiLU → ks=3 → LN → Drop → SiLU]
      Decoder:  transposed up  →  cat(skip)  →  [ks=1(2ch→ch) → LN → FiLM → SiLU
                                                 → ks=3 → LN → Drop → SiLU]
      Bottleneck: one_layers × [ks=3 → LN → FiLM → SiLU]
    """

    def __init__(self, channels, unet_depth=2, time_emb=128, one_layers=2,
                 first_ks=3, in_channels=1, out_channels=1, dropout=0.01,
                 coord_features='none', coord_h_ref=30.0, coord_xy_ref=51.0):
        super().__init__()
        self.time_emb = time_emb
        self.unet_depth = unet_depth

        # Absolute-position input channels (CoordConv) — see grid_coord_features.
        # Stored as attributes so a pickled model reproduces them at inference.
        self.coord_features = coord_features
        self.coord_h_ref = coord_h_ref
        self.coord_xy_ref = coord_xy_ref
        n_coord = _COORD_DIMS[coord_features]

        temb_dim = channels * 4
        self.time_mlp = nn.Sequential(
            GaussianFourierProjection(time_emb),
            nn.Linear(time_emb, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )

        # Time is conditioned via FiLM, not concatenation → stem sees the data
        # channels plus the (non-diffused) coordinate channels.
        self.input_conv = nn.Sequential(
            fvnn.SparseConv3d(in_channels + n_coord, channels, kernel_size=first_ks, stride=1),
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
            self.encoder_blocks.append(FiLMEncoderBlock(channels, temb_dim, dropout))

        self.bottleneck = FiLMBottleneckBlock(channels, temb_dim, one_layers)

        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        for _ in range(unet_depth):
            self.decoder_ups.append(
                fvnn.SparseConv3d(channels, channels, kernel_size=2, stride=2, transposed=True)
            )
            self.decoder_blocks.append(FiLMDecoderBlock(channels, temb_dim, dropout))

        self.output_conv = fvnn.SparseConv3d(channels, out_channels, kernel_size=1, stride=1)

    def _cat(self, x: fvnn.VDBTensor, skip: fvnn.VDBTensor) -> fvnn.VDBTensor:
        merged = torch.cat([x.data.jdata, skip.data.jdata], dim=-1)
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(merged))

    def forward(self, x, t, cond=None):
        # t arrives per-voxel (constant within a sample); collapse to one value
        # per sample, embed, then broadcast back via jidx inside each FiLM layer.
        jidx_in = x.data.jidx.long()
        t_per_sample = torch.zeros(x.grid.grid_count, device=t.device, dtype=t.dtype)
        t_per_sample[jidx_in] = t.to(t_per_sample.dtype)
        temb = self.time_mlp(t_per_sample)               # (B, temb_dim)

        # Append absolute-position channels before the stem (not diffused; derived
        # fresh from the grid so train and inference are identical).
        coords = grid_coord_features(
            x.grid, x.data.jidx, getattr(self, 'coord_features', 'none'),
            getattr(self, 'coord_h_ref', 30.0), getattr(self, 'coord_xy_ref', 51.0))
        if coords is not None:
            coords = coords.to(x.data.jdata.dtype)
            x = fvnn.VDBTensor(
                x.grid, x.grid.jagged_like(torch.cat([x.data.jdata, coords], dim=-1)))

        x = self.input_conv(x)

        skips = []
        for down, block in zip(self.encoder_downs, self.encoder_blocks):
            skips.append(x)
            x = down(x)
            x = block(x, temb)

        x = self.bottleneck(x, temb)

        for up, block, skip in zip(self.decoder_ups, self.decoder_blocks, reversed(skips)):
            x = up(x, out_grid=skip.grid)
            x = self._cat(x, skip)
            x = block(x, temb)

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
