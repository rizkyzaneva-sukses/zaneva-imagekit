import os
from PIL import Image, ImageDraw

os.makedirs("E:/Github/zaneva-imagekit/static", exist_ok=True)

sizes = [16, 32, 48, 180, 192, 512]
images = []

for size in sizes:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 2 - 1
    # Outer circle (deep blue)
    draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(37, 99, 235))
    # Mid circle (bright blue)
    ir = int(r * 0.78)
    draw.ellipse([cx-ir, cy-ir, cx+ir, cy+ir], fill=(59, 130, 246))
    # Dark lens center
    lr = int(r * 0.52)
    draw.ellipse([cx-lr, cy-lr, cx+lr, cy+lr], fill=(30, 64, 175))
    # White highlight
    hr = max(1, int(r * 0.16))
    hx, hy = cx - int(r * 0.2), cy - int(r * 0.22)
    draw.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=(255, 255, 255, 210))
    # Cross sparkle top-right
    if size >= 32:
        ll = int(r * 0.18)
        w = max(1, size // 30)
        sx, sy = cx + int(r * 0.38), cy - int(r * 0.38)
        draw.line([(sx, sy-ll), (sx, sy+ll)], fill=(255, 255, 255, 190), width=w)
        draw.line([(sx-ll, sy), (sx+ll, sy)], fill=(255, 255, 255, 190), width=w)
    images.append(img)

# Save .ico (multi-size)
images[0].save("E:/Github/zaneva-imagekit/static/favicon.ico", format="ICO",
               append_images=images[1:3])
# Save individual PNGs
for img, sz in zip(images, sizes):
    img.save("E:/Github/zaneva-imagekit/static/favicon-{}.png".format(sz))
# Apple touch icon
images[4].save("E:/Github/zaneva-imagekit/static/apple-touch-icon.png")

print("Done!")
for f in sorted(os.listdir("E:/Github/zaneva-imagekit/static/")):
    if "favicon" in f or "apple" in f:
        fp = os.path.join("E:/Github/zaneva-imagekit/static", f)
        sz = os.path.getsize(fp)
        print("  {}  ({} bytes)".format(f, sz))
