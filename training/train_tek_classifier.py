"""Second stage: classify a Tektronix crop as TDS 2014 or TDS 1002 (MobileNetV3, ImageNet-pretrained).

    .venv/Scripts/python training/train_tek_classifier.py [--epochs 15] [--arch small|large] [--size 224]

Data: datasets/tek_crops (tools/build_tek_crops.py). The model is chosen on the real held-out crops (val).
Writes models/deploy/tek_classifier.pth (weights + class names + input size).
The ImageNet weights are downloaded once by torchvision when training; the server never downloads anything.
"""
import argparse
import io
import os
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "tek_crops")
CLASSES = ["tek_tds2014", "tek_tds1002"]
SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def letterbox(img, size=SIZE):
    """Keep the aspect ratio (front panels are wide) and pad to a square."""
    w, h = img.size
    s = size / max(w, h)
    img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    out = Image.new("RGB", (size, size), (114, 114, 114))
    out.paste(img, ((size - img.size[0]) // 2, (size - img.size[1]) // 2))
    return out


class RandomBlur:
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.6))) if random.random() < 0.3 else img


class VideoDegrade:
    """Look like a distant scope in a headset / phone video frame: few pixels, compression, motion blur.

    The errors on the unseen lab were exactly these: small, soft, angled TDS 2014 fronts called TDS 1002.
    """
    def __call__(self, img):
        if random.random() < 0.5:
            side = random.randint(40, 160)
            s = side / max(img.size)
            if s < 1:
                small = img.resize((max(8, int(img.width * s)), max(8, int(img.height * s))), Image.BILINEAR)
                img = small.resize(img.size, Image.BILINEAR)
        if random.random() < 0.25:
            k = random.choice([3, 5, 7])
            img = img.filter(ImageFilter.BoxBlur(k / 3))
        if random.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=random.randint(25, 85))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return img


def eval_tf():
    return transforms.Compose([transforms.Lambda(lambda im: letterbox(im, SIZE)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_tf():
    # no horizontal flip: the button layout and labels are part of what tells the two models apart
    return transforms.Compose([
        # small hue changes only: the coloured channel knobs of the TDS 2014 are a key difference
        transforms.RandomResizedCrop(SIZE, scale=(0.55, 1.0), ratio=(0.6, 2.2)),
        transforms.RandomPerspective(0.3, p=0.4, fill=114),
        transforms.RandomRotation(12, fill=114),
        transforms.ColorJitter(0.35, 0.35, 0.3, 0.03),
        RandomBlur(),
        VideoDegrade(),
        transforms.Lambda(lambda im: letterbox(im, SIZE)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class Crops(Dataset):
    def __init__(self, split, tf):
        self.items = [(os.path.join(DATA, split, c, f), i) for i, c in enumerate(CLASSES)
                      for f in sorted(os.listdir(os.path.join(DATA, split, c)))]
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def build_model(arch="large", pretrained=True):
    if arch == "large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
    else:
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    return m


@torch.no_grad()
def evaluate(m, loader):
    m.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(torch.softmax(m(x), 1))
        ys.append(y)
    p, y = torch.cat(probs), torch.cat(ys)
    per = [(p.argmax(1)[y == k] == k).float().mean().item() for k in range(len(CLASSES))]
    acc = sum(per) / len(per)      # balanced: the small val set has twice as many TDS 2014 crops
    nll = nn.functional.nll_loss(torch.log(p + 1e-9), y).item()
    return acc, per, p, y, nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    # small/224 is the deployed one: the large/256 variant scored 98% on the real val crops but did worse on the
    # unseen lab (18/34 vs 28/34 TDS 2014 views) - it learns the training sessions, not the channel count
    ap.add_argument("--arch", default="small", choices=["large", "small"])
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--real-weight", type=float, default=3.0, help="sampling weight of real crops vs generated")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "deploy", "tek_classifier.pth"))
    ap.add_argument("--data", default=None, help="crop folder with train/ and val/ (default datasets/tek_crops)")
    ap.add_argument("--init", default=None, help="continue from this classifier checkpoint (fine-tuning)")
    ap.add_argument("--lr-scale", type=float, default=1.0)
    args = ap.parse_args()
    global SIZE, DATA
    SIZE, DATA = args.size, args.data or DATA
    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(args.threads)
    tr = Crops("train", train_tf())
    va = Crops("val", eval_tf())
    # balance the two classes per batch
    # and give real crops more weight than generated ones
    counts = np.bincount([y for _, y in tr.items], minlength=len(CLASSES))
    weights = [(args.real_weight if os.path.basename(p).startswith("real_") else 1.0) / counts[y] for p, y in tr.items]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(tr), replacement=True)
    tl = DataLoader(tr, batch_size=32, sampler=sampler, num_workers=0)
    vl = DataLoader(va, batch_size=64, num_workers=0)
    m = build_model(args.arch, pretrained=args.init is None)
    if args.init:
        m.load_state_dict(torch.load(args.init, map_location="cpu")["model"])
    opt = torch.optim.AdamW([{"params": m.features.parameters(), "lr": 2e-4 * args.lr_scale},
                             {"params": m.classifier.parameters(), "lr": 1e-3 * args.lr_scale}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, out = (-1, 1e9), args.out
    print(f"train {len(tr)} crops {dict(zip(CLASSES, counts))}, val (real) {len(va)}", flush=True)
    for ep in range(1, args.epochs + 1):
        m.train()
        tot = 0
        for x, y in tl:
            opt.zero_grad()
            loss = loss_fn(m(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        sched.step()
        acc, per, _, _, nll = evaluate(m, vl)
        mark = ""
        # best balanced accuracy; on a tie the better calibrated one (lower loss)
        if acc > best[0] or (acc == best[0] and nll < best[1]):
            best = (acc, nll)
            torch.save({"model": m.state_dict(), "classes": CLASSES, "size": SIZE, "mean": MEAN, "std": STD,
                        "arch": args.arch, "val_acc": acc}, out)
            mark = " (best, saved)"
        print(f"epoch {ep}/{args.epochs} loss {tot / len(tr):.3f} | real val acc {acc:.3f} nll {nll:.3f} "
              f"(2014 {per[0]:.2f}, 1002 {per[1]:.2f}){mark}", flush=True)
    print("best real val accuracy:", round(best[0], 3), "->", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
