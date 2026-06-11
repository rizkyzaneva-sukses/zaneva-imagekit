"""Konversi bobot RealESRGAN .pth (resmi) -> .onnx, sekali jalan saat development.

Jalankan dengan venv terpisah yang berisi torch (torch TIDAK dibutuhkan saat runtime):
    .venv-convert\\Scripts\\python.exe convert_to_onnx.py

Arsitektur RRDBNet disalin dari basicsr (Apache-2.0) agar tidak perlu
meng-install basicsr yang sudah tidak di-maintain.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).parent / "models"


# ─── RRDBNet (disalin dari basicsr.archs.rrdbnet_arch, Apache-2.0) ───

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                 num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ─── Konversi + verifikasi ───

def convert(model_name: str, scale: int):
    pth = MODELS_DIR / f"{model_name}.pth"
    out = MODELS_DIR / f"{model_name}.onnx"
    print(f"\n=== {model_name} (scale={scale}) ===")

    net = RRDBNet(scale=scale)
    state = torch.load(pth, map_location="cpu", weights_only=True)
    state = state.get("params_ema") or state.get("params") or state
    net.load_state_dict(state, strict=True)
    net.eval()

    dummy = torch.rand(1, 3, 64, 64)
    torch.onnx.export(
        net, dummy, str(out),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}},
        opset_version=17,
        dynamo=False,  # exporter legacy: stabil untuk RRDBNet + dynamic axes
    )
    print(f"  exported: {out.name} ({out.stat().st_size // 1024 // 1024}MB)")

    # Verifikasi paritas torch vs onnxruntime di 2 ukuran berbeda
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for size in [(64, 64), (100, 80)]:
        x = torch.rand(1, 3, *size)
        with torch.no_grad():
            ref = net(x).numpy()
        got = sess.run(None, {"input": x.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        assert diff < 1e-4, f"Paritas gagal {size}: max diff {diff}"
        print(f"  paritas OK {size[0]}x{size[1]}: max diff {diff:.2e}")


if __name__ == "__main__":
    convert("RealESRGAN_x4plus", 4)
    convert("RealESRGAN_x2plus", 2)
    print("\n=== Konversi selesai ===")
