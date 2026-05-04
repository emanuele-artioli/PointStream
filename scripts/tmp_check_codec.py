from src.encoder.video_io import ensure_ffmpeg_encoder_available

for codec in ["libsvtav1", "definitely_not_a_codec"]:
    try:
        ensure_ffmpeg_encoder_available(codec)
        print(codec, "OK")
    except Exception as exc:
        print(codec, type(exc).__name__, str(exc))
