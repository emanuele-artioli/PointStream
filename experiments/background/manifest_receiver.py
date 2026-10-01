"""Fresh fixed-region control receiver: manifest and charged native streams only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

def decode_manifest(manifest_path, ffmpeg="/opt/local/bin/ffmpeg"):
    manifest=json.loads(Path(manifest_path).read_text())
    width,height=manifest["geometry"]
    output=np.empty((manifest["frames"],height,width),dtype=np.uint8)
    cursor=0; receipts=[]
    for packet in manifest["packets"]:
        index=packet["frame"]; end=packet.get("hold_until",packet.get("end"))
        if index!=cursor or not index<end<=len(output):raise ValueError("packet placement gap/overlap/out-of-range")
        identity=packet["stream"]; stream=Path(identity["path"])
        payload=stream.read_bytes()
        if len(payload)!=identity["bytes"] or hashlib.sha256(payload).hexdigest()!=identity["sha256"]:
            raise ValueError("charged native payload identity mismatch")
        cmd=[ffmpeg,"-v","error","-threads","4","-i",str(stream),"-pix_fmt","yuv420p","-f","rawvideo","pipe:1"]
        result=subprocess.run(cmd,check=True,capture_output=True,timeout=180)
        expected=1 if "hold_until" in packet else end-index
        decoded=np.frombuffer(result.stdout,dtype=np.uint8).reshape(expected,width*height*3//2)
        y=decoded[:,:width*height].reshape(expected,height,width)
        output[index:end]=y[0] if "hold_until" in packet else y
        receipts.append({"frame":index,"end":end,"stream_sha256":identity["sha256"],"decoded_frames":expected,
                         "decoded_luma_sha256":hashlib.sha256(y.tobytes()).hexdigest(),"command":cmd})
        cursor=end
    if cursor!=len(output):raise ValueError("receiver frame coverage incomplete")
    return output,receipts

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args();y,receipts=decode_manifest(a.manifest)
    np.save(a.output,y)
    a.receipt.write_text(json.dumps({"manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        "receiver_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"shape":list(y.shape),
        "output_luma_sha256":hashlib.sha256(y.tobytes()).hexdigest(),"packets":receipts},indent=2)+"\n")

if __name__=="__main__":main()
