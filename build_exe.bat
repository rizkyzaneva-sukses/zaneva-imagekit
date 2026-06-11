@echo off
REM Build Zaneva ImageKit menjadi exe Windows (onedir).
REM Hasil: dist\ZanevaImageKit\ZanevaImageKit.exe
REM Catatan: tab Upscale (torch/Real-ESRGAN) tidak dibundel — lihat zaneva-imagekit.spec.

if not exist .venv (
    py -3.13 -m venv .venv
    .venv\Scripts\python.exe -m pip install "flask>=3.0.0" "rembg[cpu]>=2.0.0" ^
        "pillow>=10.0.0" "pillow-heif>=0.16.0" "numpy>=1.24.0" ^
        "opencv-python-headless>=4.8.0" "python-dotenv>=1.0.0" pyinstaller
)

.venv\Scripts\pyinstaller.exe zaneva-imagekit.spec --noconfirm
echo.
echo Selesai. Jalankan: dist\ZanevaImageKit\ZanevaImageKit.exe
