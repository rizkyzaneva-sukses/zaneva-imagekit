"""
modules/converter.py
Konversi format gambar (terutama foto iPhone HEIC/HEIF -> JPG/PNG/WEBP).

Fitur:
- Auto-orientasi berdasarkan tag EXIF Orientation (foto portrait iPhone tidak miring)
- Konversi profil warna ICC (misal Apple Display P3 -> sRGB) agar warna tidak pucat
- Opsi pertahankan atau hapus metadata EXIF (GPS, info perangkat)
- Penanganan transparansi otomatis saat konversi ke JPG
"""
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps

try:
    from PIL import ImageCms
    _CMS_OK = True
except Exception:  # pragma: no cover
    _CMS_OK = False

DEFAULT_QUALITY = 92
SUPPORTED_TARGETS = {
    "jpg":  {"pil_format": "JPEG", "ext": ".jpg",  "label": "JPG"},
    "jpeg": {"pil_format": "JPEG", "ext": ".jpg",  "label": "JPG"},
    "png":  {"pil_format": "PNG",  "ext": ".png",  "label": "PNG"},
    "webp": {"pil_format": "WEBP", "ext": ".webp", "label": "WEBP"},
}


def _convert_to_srgb(img: Image.Image) -> Image.Image:
    """Konversi gambar dengan embedded ICC profile (mis. Display P3 iPhone) ke sRGB."""
    icc_bytes = img.info.get("icc_profile")
    if not icc_bytes or not _CMS_OK:
        return img
    try:
        src_profile = ImageCms.ImageCmsProfile(BytesIO(icc_bytes))
        dst_profile = ImageCms.createProfile("sRGB")
        target_mode = "RGBA" if img.mode == "RGBA" else "RGB"
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert(target_mode)
        converted = ImageCms.profileToProfile(
            img, src_profile, dst_profile, outputMode=target_mode
        )
        if converted is not None:
            converted.info.pop("icc_profile", None)
            return converted
    except Exception:
        # Jika profil tidak standar atau gagal dikonversi, lanjutkan dengan gambar asli
        pass
    return img


def _parse_hex(hex_str: str) -> tuple:
    """Parse hex color string ke RGB tuple."""
    hex_str = (hex_str or "#FFFFFF").lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    try:
        return tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


def _format_size(num_bytes: int) -> str:
    """Format ukuran byte ke KB atau MB yang mudah dibaca."""
    kb = num_bytes / 1024
    if kb >= 1024:
        return f"{kb / 1024:.2f}MB"
    return f"{kb:.0f}KB"


def process_image(
    in_path: Path,
    out_dir: Path,
    target_format: str = "jpg",
    quality: int = DEFAULT_QUALITY,
    keep_exif: bool = False,
    bg_color: str = "#FFFFFF",
) -> dict:
    """
    Konversi satu gambar ke format target (jpg / png / webp).
    Mengembalikan dict info status, output_id, resolusi, dan ukuran sebelum/sesudah.
    """
    try:
        fmt_key = (target_format or "jpg").lower().strip(".")
        if fmt_key not in SUPPORTED_TARGETS:
            fmt_key = "jpg"
        target_cfg = SUPPORTED_TARGETS[fmt_key]
        pil_format = target_cfg["pil_format"]
        out_ext = target_cfg["ext"]
        target_label = target_cfg["label"]

        quality = max(10, min(100, int(quality or DEFAULT_QUALITY)))
        size_before = in_path.stat().st_size

        img = Image.open(in_path)
        orig_format = (img.format or in_path.suffix.lstrip(".")).upper()
        if orig_format in ("HEIF", "HEIC"):
            orig_format = "HEIC"
        elif orig_format == "JPEG":
            orig_format = "JPG"

        # 1. Perbaiki rotasi otomatis berdasarkan EXIF Orientation (penting untuk foto iPhone)
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        # Ambil EXIF setelah transpose (Orientation sudah di-reset ke 1 oleh Pillow)
        exif_bytes = None
        if keep_exif:
            try:
                exif_obj = img.getexif()
                if exif_obj:
                    exif_bytes = exif_obj.tobytes()
            except Exception:
                exif_bytes = img.info.get("exif")

        # 2. Cek apakah gambar memiliki transparansi
        has_transparency = (
            img.mode in ("RGBA", "LA")
            or (img.mode == "P" and "transparency" in img.info)
        )

        if has_transparency:
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        # 3. Konversi profil warna (mis. Display P3 -> sRGB) agar warna akurat di semua aplikasi
        img = _convert_to_srgb(img)

        # 4. Jika target JPEG dan gambar punya alpha, flatten di atas background solid
        if pil_format == "JPEG" and img.mode == "RGBA":
            bg_rgb = _parse_hex(bg_color)
            canvas = Image.new("RGB", img.size, bg_rgb)
            canvas.paste(img, mask=img.split()[3])
            img = canvas
        elif pil_format == "JPEG" and img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        out_name = f"{in_path.stem}_conv{out_ext}"
        out_path = out_dir / out_name

        save_kwargs = {}
        if pil_format == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
            # Subsampling 4:4:4 untuk kualitas >= 90 agar detail & warna tajam
            if quality >= 90:
                save_kwargs["subsampling"] = 0
        elif pil_format == "WEBP":
            save_kwargs["quality"] = quality
            save_kwargs["method"] = 4
        elif pil_format == "PNG":
            save_kwargs["optimize"] = True

        if exif_bytes:
            save_kwargs["exif"] = exif_bytes

        img.save(out_path, format=pil_format, **save_kwargs)
        size_after = out_path.stat().st_size

        return {
            "status": "ok",
            "output_id": out_name,
            "orig_format": orig_format,
            "target_format": target_label,
            "resolution": f"{w}×{h}",
            "size_before_str": _format_size(size_before),
            "size_after_str": _format_size(size_after),
            "size_kb": round(size_after / 1024, 1),
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
