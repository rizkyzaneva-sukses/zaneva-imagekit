# -*- mode: python ; coding: utf-8 -*-
# Build: .venv\Scripts\pyinstaller.exe zaneva-imagekit.spec --noconfirm
# Hasil: dist\ZanevaImageKit\ZanevaImageKit.exe (mode onedir)
#
# Tab Upscale (torch/Real-ESRGAN) sengaja TIDAK dibundel agar ukuran exe
# tetap wajar — sama seperti image Docker core. Remove BG & Resize full jalan.
from PyInstaller.utils.hooks import collect_submodules, copy_metadata

# rembg memuat session model secara dinamis (scan folder), jadi seluruh
# submodule harus dipaksa ikut ke bundle.
hiddenimports = collect_submodules("rembg") + ["pillow_heif"]

datas = [
    ("templates", "templates"),
    ("models/RealESRGAN_x4plus.onnx", "models"),
    ("models/RealESRGAN_x2plus.onnx", "models"),
]
# Beberapa package membaca versinya sendiri via importlib.metadata saat
# runtime, jadi metadata-nya wajib ikut dibundel.
for pkg in ("rembg", "pymatting", "pooch", "pillow_heif"):
    datas += copy_metadata(pkg)

a = Analysis(
    ["desktop.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "torchvision", "basicsr", "realesrgan", "tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ZanevaImageKit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # console dibiarkan tampil: log + progress download model
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ZanevaImageKit",
)
