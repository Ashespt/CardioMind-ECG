"""Build the ZS manifest from local DAT files; no labels enter inference."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dat-dir",type=Path,default=Path("data/dat"))
    p.add_argument("--image-dir",type=Path,default=Path("data/images"))
    p.add_argument("--output",type=Path,default=Path("data/manifest.jsonl"))
    a=p.parse_args()
    files=sorted(a.dat_dir.glob("*.dat"))
    if not files: raise SystemExit("No DAT files found; supply real records under data/dat/")
    rows=[]
    for f in files:
        image=a.image_dir/(f.stem+".png")
        if not image.is_file(): raise FileNotFoundError(image)
        rows.append({"id":"zs/"+f.stem,"subset":"zs","image_path":str(image)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(r)+"\n" for r in rows))
    print(f"Wrote {len(rows)} records to {a.output}")
if __name__=="__main__": main()
