"""Bundle everything needed to train on the NVIDIA workstation into one zip (stays on the lab network).

    .venv/Scripts/python tools/package_for_workstation.py            -> dist/workstation_<date>.zip

Contents: the code (git-tracked files), the built dataset (datasets/oscilloscopes3), the starting weights
(models/deploy + COCO checkpoint) and WORKSTATION.md with the exact commands. No virtual environments,
no raw photos (the dataset is already built from them). Copy the zip by USB or the lab network share.
"""
import datetime
import hashlib
import os
import subprocess
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUIDE = """# Training on the NVIDIA workstation

1. Unzip, open a terminal in the folder, create the environment (Python 3.10-3.12, CUDA PyTorch):
       python -m venv .venv
       .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match the CUDA driver
       .venv/bin/pip install -r requirements-deploy.txt
       git clone --depth 1 --branch 0.3.0 https://github.com/Megvii-BaseDetection/YOLOX.git models/yolox
   (Windows: use .venv\\Scripts\\ instead of .venv/bin/)

2. Train (the GPU is used automatically; about 5-15 minutes instead of 2 hours on the CPU):
       .venv/bin/python training/train_cpu.py --exp yolox_tiny_osc3 --epochs 30 --batch 16 \\
           --init models/deploy/yolox_tiny_osc3.pth

   Or the official YOLOX trainer with the same experiment file:
       PYTHONPATH=models/yolox:training .venv/bin/python models/yolox/tools/train.py \\
           -f training/yolox_tiny_osc3.py -d 1 -b 16 -c models/deploy/yolox_tiny_osc3.pth

3. Score it on the real held-out photos (same test as on the lab PC):
       .venv/bin/python tools/eval_app.py models/training/yolox_tiny_osc3/best_ckpt.pth

4. Bring models/training/yolox_tiny_osc3/best_ckpt.pth back to the lab PC. If it beats the deployed model,
   copy it to models/deploy/yolox_tiny_osc3.pth, commit and push (Render redeploys automatically).
"""


def main():
    stamp = datetime.date.today().isoformat()
    out_dir = os.path.join(ROOT, "dist")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"workstation_{stamp}.zip")

    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    extra_dirs = ["datasets/oscilloscopes3"]
    extra_files = ["models/yolox_tiny.pth", "raw/real_labels.json"]
    for d in extra_dirs:
        for base, _, names in os.walk(os.path.join(ROOT, d)):
            files += [os.path.relpath(os.path.join(base, n), ROOT).replace(os.sep, "/") for n in names]
    files += [f for f in extra_files if os.path.exists(os.path.join(ROOT, f))]

    sha = hashlib.sha256()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("WORKSTATION.md", GUIDE)
        for f in sorted(set(files)):
            p = os.path.join(ROOT, f)
            if os.path.isfile(p):
                z.write(p, f)
                sha.update(f.encode())
    size = os.path.getsize(out) / 2 ** 20
    print(f"{out}  ({len(set(files))} files, {size:.0f} MB)")
    print("Open WORKSTATION.md inside the zip for the commands.")


if __name__ == "__main__":
    main()
