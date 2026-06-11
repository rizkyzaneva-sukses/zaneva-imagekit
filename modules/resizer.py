from pathlib import Path
from PIL import Image, ImageOps
import zipfile
from io import BytesIO

# Platform presets: name -> (width, height, label)
PLATFORM_PRESETS = {
    "shopee":      (1500, 1500, "Shopee Produk"),
    "tiktok_feed": (1080, 1920, "TikTok Feed"),
    "tiktok_thumb":(1080, 1080, "TikTok Thumbnail"),
    "ig_feed":     (1080, 1350, "Instagram Feed"),
    "ig_square":   (1080, 1080, "Instagram Square"),
    "marketplace": (1200,  675, "Marketplace Banner"),
    "original":    (None,  None, "Original + Compress"),
}

DEFAULT_QUALITY = 85


def _crop_center(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Crop image from center to target aspect ratio, then resize."""
    src_w, src_h = img.size
    src_ratio = src_w / src_h
    tgt_ratio = target_w / target_h

    # Auto-switch to fit+padding if crop ratio is too extreme (>3:1)
    if max(tgt_ratio, 1/tgt_ratio) / max(src_ratio, 1/src_ratio) > 3:
        return _fit_padding(img, target_w, target_h, (255, 255, 255))

    if src_ratio > tgt_ratio:
        # Image wider than target — crop sides
        new_w = int(src_h * tgt_ratio)
        left = (src_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, src_h))
    else:
        # Image taller than target — crop top/bottom
        new_h = int(src_w / tgt_ratio)
        top = (src_h - new_h) // 2
        img = img.crop((0, top, src_w, top + new_h))

    # Always output the exact preset size (upscale kecil bila perlu) agar
    # dimensi hasil sesuai label platform (mis. Shopee 1500×1500).
    img = img.resize((target_w, target_h), Image.LANCZOS)

    return img


def _fit_padding(img: Image.Image, target_w: int, target_h: int,
                 pad_color: tuple) -> Image.Image:
    """Fit image within target size, pad remainder with pad_color."""
    src_w, src_h = img.size

    # Skala agar muat di dalam target (boleh memperbesar foto kecil).
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    offset_x = (target_w - new_w) // 2
    offset_y = (target_h - new_h) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def _parse_hex(hex_str: str) -> tuple:
    """Parse hex color string to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c*2 for c in hex_str)
    try:
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return (255, 255, 255)


def process_image(
    in_path: Path,
    out_dir: Path,
    presets: list,
    method: str = "crop",
    pad_color: str = "#FFFFFF",
    quality: int = DEFAULT_QUALITY,
    original_name: str = None,
) -> dict:
    """
    Resize + compress a single image for multiple presets.
    Returns {status, outputs: [{preset, output_id, size_kb}]}
    """
    try:
        img = Image.open(in_path)
        has_transparency = img.mode in ("RGBA", "LA") or \
            (img.mode == "P" and "transparency" in img.info)

        stem = original_name or in_path.stem
        pad_rgb = _parse_hex(pad_color)
        outputs = []

        for preset_key in presets:
            if preset_key not in PLATFORM_PRESETS:
                continue

            target_w, target_h, label = PLATFORM_PRESETS[preset_key]

            if preset_key == "original":
                # Just compress, no resize
                work_img = img.copy()
                if work_img.mode in ("RGBA", "LA", "P"):
                    if has_transparency:
                        out_name = f"{stem}_original.png"
                        out_path = out_dir / out_name
                        work_img.save(out_path, format="PNG", optimize=True)
                    else:
                        work_img = work_img.convert("RGB")
                        out_name = f"{stem}_original.jpg"
                        out_path = out_dir / out_name
                        work_img.save(out_path, format="JPEG",
                                      quality=quality, optimize=True)
                else:
                    work_img = work_img.convert("RGB")
                    out_name = f"{stem}_original.jpg"
                    out_path = out_dir / out_name
                    work_img.save(out_path, format="JPEG",
                                  quality=quality, optimize=True)
            else:
                # Resize
                work_img = img.copy()

                # Flatten transparency before resize (except if input has alpha)
                if work_img.mode in ("RGBA", "LA", "P"):
                    if has_transparency:
                        # Keep as RGBA for proper paste
                        work_img = work_img.convert("RGBA")
                    else:
                        work_img = work_img.convert("RGB")
                else:
                    work_img = work_img.convert("RGB")

                if method == "crop":
                    resized = _crop_center(work_img.convert("RGB"), target_w, target_h)
                else:
                    resized = _fit_padding(work_img.convert("RGB"),
                                          target_w, target_h, pad_rgb)

                # Output format
                if has_transparency:
                    out_name = f"{stem}_{preset_key}.png"
                    out_path = out_dir / out_name
                    resized.save(out_path, format="PNG", optimize=True)
                else:
                    out_name = f"{stem}_{preset_key}.jpg"
                    out_path = out_dir / out_name
                    resized.save(out_path, format="JPEG",
                                 quality=quality, optimize=True)

            size_kb = round(out_path.stat().st_size / 1024, 1)
            outputs.append({
                "preset": preset_key,
                "label": label,
                "output_id": out_name,
                "size_kb": size_kb,
            })

        try:
            in_path.unlink(missing_ok=True)
        except OSError:
            pass  # input masih dipakai (preview di Windows); dibersihkan auto-cleanup
        return {"status": "ok", "outputs": outputs}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def build_zip(output_ids: list, out_dir: Path, stem_map: dict) -> BytesIO:
    """
    Build ZIP with folder per original file.
    stem_map: {output_id -> original_stem}
    """
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for oid in output_ids:
            p = out_dir / oid
            if p.exists():
                folder = stem_map.get(oid, "misc")
                zf.write(p, arcname=f"{folder}/{oid}")
    buf.seek(0)
    return buf
