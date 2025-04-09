from PIL import Image
import zstandard as zstd

def png_to_jpeg(png_path, jpeg_path, quality=75):
    """
    Convert PNG to JPEG with adjustable compression.

    Args:
        png_path (str): Input PNG file path.
        jpeg_path (str): Output JPEG file path.
        quality (int): JPEG quality (1 = worst, 95 = best). Default is 75.
    """
    # Open PNG and convert to RGB (JPEG doesn't support alpha)
    with Image.open(png_path) as img:
        rgb_img = img.convert('RGB')
        rgb_img.save(jpeg_path, format='JPEG', quality=quality, optimize=True)

def delta_encode(data):
    return [data[0]] + [data[i] - data[i-1] for i in range(1, len(data))]

def compress_with_zstd(data):
    cctx = zstd.ZstdCompressor()
    return cctx.compress(data)

if __name__ == "__main__":
    # Compress PNG into JPEG
    png_to_jpeg("010-modified.png", "010-100.jpg", quality=30)
