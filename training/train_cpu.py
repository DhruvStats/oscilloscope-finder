"""CPU fine-tuning of YOLOX-Tiny for machines without CUDA.

The official YOLOX trainer requires CUDA. This loop uses the same Exp, data
pipeline (mosaic, flips, HSV), loss and LR schedule, but runs on the CPU and
evaluates COCO AP with pycocotools after every `eval_interval` epochs.

    .venv/Scripts/python training/train_cpu.py [--exp yolox_tiny_osc3] [--epochs N] [--batch 8] [--init ckpt.pth]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models", "yolox"))
sys.path.insert(0, os.path.join(ROOT, "training"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from yolox.data.data_augment import ValTransform  # noqa: E402
from yolox.utils import load_ckpt, postprocess  # noqa: E402

import importlib  # noqa: E402


def evaluate(model, exp, split="val", conf=0.01):
    ann = os.path.join(exp.data_dir, "annotations", f"instances_{split}2017.json")
    gt = COCO(ann)
    tf = ValTransform(legacy=False)
    dets = []
    model.eval()
    with torch.no_grad():
        for img_id in gt.getImgIds():
            info = gt.loadImgs(img_id)[0]
            img = cv2.imread(os.path.join(exp.data_dir, f"{split}2017", info["file_name"]))
            ratio = min(exp.test_size[0] / img.shape[0], exp.test_size[1] / img.shape[1])
            x, _ = tf(img, None, exp.test_size)
            out = postprocess(model(torch.from_numpy(x).unsqueeze(0).float()), exp.num_classes, conf, exp.nmsthre)[0]
            if out is None:
                continue
            for x0, y0, x1, y1, obj, cls_conf, cls in out.numpy():
                b = np.array([x0, y0, x1, y1]) / ratio
                dets.append({"image_id": img_id, "category_id": int(cls) + 1, "score": float(obj * cls_conf),
                             "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]})
    model.train()
    if not dets:
        return 0.0, 0.0
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    # precision[iou, recall, class, area=all, maxDets=100]; iou index 0 is IoU 0.50
    for k, cat in enumerate(gt.loadCats(gt.getCatIds())):
        p50, p = ev.eval["precision"][0, :, k, 0, 2], ev.eval["precision"][:, :, k, 0, 2]
        print(f"  {cat['name']:<14} AP {p[p > -1].mean():.3f}  AP50 {p50[p50 > -1].mean():.3f}", flush=True)
    return float(ev.stats[0]), float(ev.stats[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--exp", default="yolox_tiny_rtb2004", help="experiment module in training/")
    ap.add_argument("--init", default=os.path.join(ROOT, "models", "yolox_tiny.pth"),
                    help="starting weights; layers whose shape differs (the class head) are skipped")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    exp = importlib.import_module(args.exp).Exp()
    if args.epochs:
        exp.max_epoch = args.epochs
    out_dir = os.path.join(exp.output_dir, exp.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    model = exp.get_model()
    ckpt = torch.load(args.init, map_location="cpu")
    load_ckpt(model, ckpt["model"])   # backbone/neck; head layers with a different class count are skipped
    model.train()

    loader = exp.get_data_loader(args.batch, is_distributed=False, no_aug=False)
    iters_per_epoch = int(np.ceil(len(exp.dataset) / args.batch))
    optimizer = exp.get_optimizer(args.batch)
    scheduler = exp.get_lr_scheduler(exp.basic_lr_per_img * args.batch, iters_per_epoch)

    log = []
    best = -1.0
    it = iter(loader)
    t0 = time.time()
    for epoch in range(exp.max_epoch):
        if epoch == exp.max_epoch - exp.no_aug_epochs:
            print("--- closing mosaic, enabling L1 loss ---", flush=True)
            loader.close_mosaic()
            model.head.use_l1 = True
            it = iter(loader)
        losses = []
        for i in range(iters_per_epoch):
            inps, targets, _, _ = next(it)
            inps, targets = inps.float(), targets.float()
            targets.requires_grad = False
            out = model(inps, targets)
            optimizer.zero_grad()
            out["total_loss"].backward()
            optimizer.step()
            lr = scheduler.update_lr(epoch * iters_per_epoch + i + 1)
            for g in optimizer.param_groups:
                g["lr"] = lr
            losses.append(float(out["total_loss"]))
        msg = f"epoch {epoch + 1}/{exp.max_epoch} loss {np.mean(losses):.3f} lr {lr:.5f} elapsed {time.time() - t0:.0f}s"
        entry = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        if (epoch + 1) % exp.eval_interval == 0 or epoch + 1 == exp.max_epoch:
            ap50_95, ap50 = evaluate(model, exp, "val")
            entry.update(val_ap=ap50_95, val_ap50=ap50)
            msg += f" | val AP {ap50_95:.3f} AP50 {ap50:.3f}"
            state = {"model": model.state_dict(), "epoch": epoch + 1, "val_ap": ap50_95}
            torch.save(state, os.path.join(out_dir, "last_ckpt.pth"))
            if ap50_95 > best:
                best = ap50_95
                torch.save(state, os.path.join(out_dir, "best_ckpt.pth"))
                msg += " (best)"
        print(msg, flush=True)
        log.append(entry)
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(log, f, indent=1)

    print("best val AP:", best, flush=True)


if __name__ == "__main__":
    main()
