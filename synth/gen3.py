"""Synthetic COCO dataset for three oscilloscopes: R&S RTB2004, Tektronix TDS 2014, Tektronix TDS 1002.

Each scope is a textured 3D box built from real photos (see tek_textures.py and the RTB2004 textures),
rendered at random poses and composited on bench scenes with real clutter cut out of the lab photos.
Boxes are exact (computed from the render, visible part only).

With --real-labels, the checked boxes on the real lab photos are used three ways:
  - train photos: each scope is cut out (GrabCut inside its box) and pasted into new scenes ("cut and paste"),
    the photos themselves join the training set, and their scope-free areas become extra backgrounds;
  - test photos: written unchanged to test2017 - real images never seen in training, for an honest score.

    .venv/Scripts/python synth/gen3.py --per-class 200 --real-labels raw/real_labels.json
"""
import argparse
import json
import os
import random
import shutil

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(HERE, 'textures')
SESSION = os.path.join(ROOT, 'raw', 'rtb_session')
W, H = 640, 480

CLASSES = ['rs_rtb2004', 'tek_tds2014', 'tek_tds1002']
REAL_WIDTH_MM = {'rs_rtb2004': 390, 'tek_tds2014': 326, 'tek_tds1002': 324}
FACES = ['front', 'back', 'left', 'right', 'top', 'bottom']

# scopes visible in the session photos: never use these areas as background or clutter
OCCUPIED = {
    1: [(101, 463, 516, 1095), (0, 560, 125, 840)],
    2: [(223, 54, 1205, 650)],
    3: [(245, 106, 1082, 579), (0, 195, 245, 530)],
    4: [(100, 440, 670, 1256)],
    5: [(206, 134, 1326, 602), (0, 90, 70, 490)],
    6: [(60, 200, 690, 1340), (0, 300, 80, 640)],
    7: [(190, 800, 612, 1050), (630, 880, 717, 1110)],
    8: [(177, 178, 1287, 636)],
}
DISTRACTORS = [
    (7, (128, 795, 197, 992)),    # water bottle
    (7, (306, 598, 347, 702)),    # drink bottle
    (7, (338, 566, 384, 692)),    # snack jar
    (7, (252, 508, 308, 694)),    # mannequin head
    (7, (458, 535, 717, 798)),    # backpack
    (7, (588, 598, 717, 757)),    # desk calendar
    (7, (273, 703, 432, 762)),    # circuit boards
    (7, (12, 683, 188, 842)),     # chair
    (7, (12, 352, 78, 472)),      # coffee machine
    (2, (0, 362, 162, 428)),      # key fob
    (3, (1128, 340, 1595, 472)),  # paper sheets
]


def load_models():
    def face(name):
        return cv2.imread(os.path.join(TEX, name + '.png'), cv2.IMREAD_UNCHANGED)

    models = {'rs_rtb2004': [], 'tek_tds2014': [], 'tek_tds1002': []}
    for variant in ('rtb_studio', 'rtb_photo'):
        models['rs_rtb2004'].append({f: face(f'{variant}_{f}') for f in FACES})
    shared = {f: face(f'tek_{f}') for f in FACES if f != 'front'}
    models['tek_tds2014'].append(dict(shared, front=face('tek_tds2014_front')))
    models['tek_tds1002'].append(dict(shared, front=face('tek_tds1002_front'), back=face('tek_tds1002_back'),
                                      top=face('tek_tds1002_top')))
    return models


def box_corners(tex):
    fw = tex['front'].shape[1]
    fh = tex['front'].shape[0]
    d = tex['left'].shape[1]
    hw, hh, hd = fw / 2, fh / 2, d / 2
    c = {
        'flt': (-hw, -hh, hd), 'frt': (hw, -hh, hd), 'frb': (hw, hh, hd), 'flb': (-hw, hh, hd),
        'blt': (-hw, -hh, -hd), 'brt': (hw, -hh, -hd), 'brb': (hw, hh, -hd), 'blb': (-hw, hh, -hd),
    }
    keys = [
        ('front', ['flt', 'frt', 'frb', 'flb'], (0, 0, 1)),
        ('back', ['brt', 'blt', 'blb', 'brb'], (0, 0, -1)),
        ('right', ['frt', 'brt', 'brb', 'frb'], (1, 0, 0)),
        ('left', ['blt', 'flt', 'flb', 'blb'], (-1, 0, 0)),
        ('top', ['blt', 'brt', 'frt', 'flt'], (0, -1, 0)),
        ('bottom', ['flb', 'frb', 'brb', 'blb'], (0, 1, 0)),
    ]
    return c, keys


def rotation(yaw, pitch, roll):
    cy_, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def render_rgba(tex, yaw, pitch, roll):
    S = 1400
    focal, dist = random.uniform(1300, 2600), random.uniform(2600, 4200)
    R = rotation(yaw, pitch, roll)
    light = np.array([random.uniform(-1, 1), random.uniform(-1, -0.2), random.uniform(0.3, 1)])
    light /= np.linalg.norm(light)
    amb, dif = random.uniform(0.72, 0.95), random.uniform(0.05, 0.3)
    corners, keys = box_corners(tex)
    col = np.zeros((S, S, 3), np.float32)
    alpha = np.zeros((S, S), np.float32)

    def proj(p):
        return S / 2 + focal * p[0] / (dist - p[2]), S / 2 + focal * p[1] / (dist - p[2])

    faces = []
    for name, ks, n in keys:
        nw = R @ np.array(n, float)
        pts3 = [R @ np.array(corners[k]) for k in ks]
        c = np.mean(pts3, axis=0)
        if nw @ (np.array([0, 0, dist]) - c) <= 0:
            continue
        faces.append((c[2], tex[name], np.float32([proj(p) for p in pts3]), amb + dif * max(0, nw @ light)))
    faces.sort(key=lambda f: f[0])
    for _, t, dst, shade in faces:
        h, w = t.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), dst)
        wp = cv2.warpPerspective(t, M, (S, S), flags=cv2.INTER_LINEAR)
        a = wp[..., 3].astype(np.float32) / 255
        col = col * (1 - a[..., None]) + wp[..., :3].astype(np.float32) * shade * a[..., None]
        alpha = np.maximum(alpha, a)
    ys, xs = np.where(alpha > 0.5)
    out = np.dstack([np.clip(col, 0, 255), alpha * 255]).astype(np.uint8)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def random_pose():
    r = random.random()
    yaw = random.uniform(0, 2 * np.pi)
    if r < 0.4:
        yaw = random.gauss(0, 0.6)      # front-facing views carry the model identity: oversample them
    if r < 0.75:
        pitch = np.radians(random.uniform(3, 40))
    elif r < 0.9:
        pitch = np.radians(random.uniform(-6, 6))
    else:
        pitch = np.radians(random.uniform(40, 75))
    return yaw, pitch, np.radians(random.gauss(0, 5))


def jitter(img, contrast, bright):
    out = img.astype(np.float32) * (1 + random.uniform(-contrast, contrast)) + random.uniform(-bright, bright)
    out *= np.array([random.uniform(0.96, 1.04) for _ in range(3)], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def grabcut(img, rect):
    x0, y0, x1, y1 = rect
    X0, Y0, X1, Y1 = max(0, x0 - 6), max(0, y0 - 6), min(img.shape[1], x1 + 6), min(img.shape[0], y1 + 6)
    crop = img[Y0:Y1, X0:X1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.grabCut(crop, mask, (x0 - X0 + 1, y0 - Y0 + 1, x1 - x0 - 2, y1 - y0 - 2),
                np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(m)
    cv2.drawContours(filled, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    filled = cv2.GaussianBlur(filled, (5, 5), 0)
    ys, xs = np.where(filled > 128)
    return np.dstack([crop, filled])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def overlaps(r, boxes, m=12):
    x0, y0, x1, y1 = r
    return any(not (x1 < bx0 - m or x0 > bx1 + m or y1 < by0 - m or y0 > by1 + m) for bx0, by0, bx1, by1 in boxes)


def random_background(photos):
    for _ in range(500):
        n = random.choice(list(photos))
        img = photos[n]
        h, w = img.shape[:2]
        cw = random.randint(220, min(w, int(h * 4 / 3), 900))
        ch = int(cw * 3 / 4)
        x, y = random.randint(0, w - cw), random.randint(0, h - ch)
        if not overlaps((x, y, x + cw, y + ch), OCCUPIED[n]):
            bg = cv2.resize(img[y:y + ch, x:x + cw], (W, H), interpolation=cv2.INTER_CUBIC)
            return jitter(cv2.flip(bg, 1) if random.random() < 0.5 else bg, 0.25, 25)
    raise RuntimeError('no free background area found')


def paste(canvas, obj, x, y):
    h, w = obj.shape[:2]
    full = np.zeros((H, W), np.float32)
    X0, Y0, X1, Y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if X1 <= X0 or Y1 <= Y0:
        return full
    o = obj[Y0 - y:Y1 - y, X0 - x:X1 - x]
    a = o[..., 3:4].astype(np.float32) / 255
    canvas[Y0:Y1, X0:X1] = (canvas[Y0:Y1, X0:X1] * (1 - a) + o[..., :3] * a).astype(np.uint8)
    full[Y0:Y1, X0:X1] = a[..., 0]
    return full


def contact_shadow(canvas, x, y_bottom, w, strength):
    sh = np.zeros((H, W), np.float32)
    cv2.ellipse(sh, (int(x + w / 2), int(y_bottom - 3)), (int(w * 0.5), max(4, int(w * 0.05))), 0, 0, 360, 1, -1)
    sh = cv2.GaussianBlur(sh, (0, 0), max(3, w * 0.03))
    canvas[:] = (canvas.astype(np.float32) * (1 - strength * sh[..., None])).astype(np.uint8)


def place_distractor(canvas, d):
    s = min(random.uniform(0.15, 0.55) * H, d.shape[0] * 1.3) / d.shape[0]
    d = cv2.resize(d, (max(2, int(d.shape[1] * s)), max(2, int(d.shape[0] * s))), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:
        d = cv2.flip(d, 1)
    d = np.dstack([jitter(d[..., :3], 0.15, 15), d[..., 3]])
    h, w = d.shape[:2]
    x = random.randint(-w // 4, W - 3 * w // 4)
    y = random.randint(int(H * 0.25) - h // 2, H - int(h * 0.8))
    contact_shadow(canvas, x, y + h, w, random.uniform(0.15, 0.35))
    return paste(canvas, d, x, y)


def finish(img):
    img = jitter(img, 0.12, 12)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (random.choice([3, 5]),) * 2, 0)
    if random.random() < 0.15:
        k = random.choice([5, 7, 9])
        kern = np.zeros((k, k), np.float32)
        kern[k // 2, :] = 1 / k
        img = cv2.filter2D(img, -1, cv2.warpAffine(kern, cv2.getRotationMatrix2D((k / 2, k / 2), random.uniform(0, 180), 1), (k, k)))
    noise = np.random.normal(0, random.uniform(0, 6), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def load_real(path):
    """real_labels.json: {"images": [{"file", "session", "split": "train"|"test", "boxes": [{"cls", "bbox": [x,y,w,h]}]}]}"""
    with open(path) as f:
        items = json.load(f)['images']
    for it in items:
        it['img'] = cv2.imread(os.path.join(ROOT, it['file']))
    return items


def real_cutouts(items):
    """Cut every sufficiently large, untruncated scope out of the train photos."""
    cuts = {c: [] for c in CLASSES}
    for it in items:
        img = it['img']
        h, w = img.shape[:2]
        for b in it['boxes']:
            x, y, bw, bh = b['bbox']
            truncated = x <= 3 or y <= 3 or x + bw >= w - 3 or y + bh >= h - 3
            if min(bw, bh) < 90 or truncated:
                continue
            try:
                c = grabcut(img, (int(x), int(y), int(x + bw), int(y + bh)))
            except ValueError:
                continue
            # a cut-out that lost most of its box is a failed segmentation
            if (c[..., 3] > 128).sum() > 0.55 * bw * bh:
                cuts[b['cls']].append(c)
    return cuts


def object_for(cls, models, cuts, cutout_frac):
    if cuts.get(cls) and random.random() < cutout_frac:
        c = random.choice(cuts[cls])
        # real photo of a real scope: never mirror (text and layout would be wrong), only tilt slightly
        ang = random.uniform(-4, 4)
        h, w = c.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
        c = cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        ys, xs = np.where(c[..., 3] > 128)
        return c[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return render_rgba(random.choice(models[cls]), *random_pose())


def make_scene(classes, models, distractors, photos, cuts=None, cutout_frac=0.0):
    canvas = random_background(photos)
    for d in random.sample(distractors, random.randint(1, 4)):
        place_distractor(canvas, d)
    px_per_mm = random.uniform(0.18, 0.75) * W / 390 / max(1, len(classes) * 0.7)
    masks = []          # (class, alpha) in paint order
    random.shuffle(classes)
    for cls in classes:
        obj = object_for(cls, models, cuts or {}, cutout_frac)
        target_w = REAL_WIDTH_MM[cls] * px_per_mm * random.uniform(0.8, 1.2)
        obj = cv2.resize(obj, (max(8, int(target_w)), max(8, int(obj.shape[0] * target_w / obj.shape[1]))),
                         interpolation=cv2.INTER_AREA)
        obj = np.dstack([jitter(obj[..., :3], 0.15, 18), obj[..., 3]])
        h, w = obj.shape[:2]
        for _ in range(30):     # find a spot where it does not hide earlier scopes too much
            truncate = random.random() < 0.1
            lo = -w // 3 if truncate else 0
            x = random.randint(lo, max(lo, W - w - lo))
            y_bottom = max(random.randint(int(H * 0.45), H + (h // 4 if truncate else 0)), h // 2)
            new = np.zeros((H, W), np.float32)
            Y0, X0 = max(0, y_bottom - h), max(0, x)
            Y1, X1 = min(H, y_bottom), min(W, x + w)
            if X1 > X0 and Y1 > Y0:
                new[Y0:Y1, X0:X1] = obj[Y0 - (y_bottom - h):Y1 - (y_bottom - h), X0 - x:X1 - x, 3] / 255
            if all(((m > 0.5) & (new < 0.5)).sum() >= 0.6 * (m > 0.5).sum() for _, m in masks):
                break
        contact_shadow(canvas, x, y_bottom, w, random.uniform(0.25, 0.45))
        a = paste(canvas, obj, x, y_bottom - h)
        masks = [(c, np.where(a > 0.5, 0, m)) for c, m in masks] + [(cls, a)]
    # clutter in front, only if every scope stays at least 60% visible
    for d in random.sample(distractors, random.randint(0, 2)):
        trial = canvas.copy()
        occ = place_distractor(trial, d)
        if all(((m > 0.5) & (occ < 0.5)).sum() >= 0.6 * (m > 0.5).sum() for _, m in masks):
            canvas = trial
            masks = [(c, np.where(occ > 0.5, 0, m)) for c, m in masks]
    anns = []
    for cls, m in masks:
        ys, xs = np.where(m > 0.5)
        if len(xs) > 200:
            anns.append((cls, [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]))
    return finish(canvas), anns


def write_split(out, split, items):
    d = os.path.join(out, f'{split}2017')
    os.makedirs(d, exist_ok=True)
    coco = {'images': [], 'annotations': [],
            'categories': [{'id': i + 1, 'name': c, 'supercategory': 'oscilloscope'} for i, c in enumerate(CLASSES)]}
    for i, (img, anns, name) in enumerate(items, 1):
        cv2.imwrite(os.path.join(d, name), img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(78, 95)])
        coco['images'].append({'id': i, 'file_name': name, 'width': img.shape[1], 'height': img.shape[0]})
        for cls, b in anns:
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': i,
                                        'category_id': CLASSES.index(cls) + 1, 'bbox': b,
                                        'area': b[2] * b[3], 'iscrowd': 0})
    os.makedirs(os.path.join(out, 'annotations'), exist_ok=True)
    with open(os.path.join(out, 'annotations', f'instances_{split}2017.json'), 'w') as f:
        json.dump(coco, f, indent=1)
    counts = {c: sum(1 for a in coco['annotations'] if a['category_id'] == i + 1) for i, c in enumerate(CLASSES)}
    return len(items), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=200)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--neg-frac', type=float, default=0.1)
    ap.add_argument('--out', default=os.path.join(ROOT, 'datasets', 'oscilloscopes3'))
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--real-labels', default=None, help='checked boxes on the real photos (see load_real)')
    ap.add_argument('--cutout-frac', type=float, default=0.6, help='share of pasted scopes taken from real photos')
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    photos = {n: cv2.imread(os.path.join(SESSION, f'{n}.jpg')) for n in OCCUPIED}
    distractors = [grabcut(photos[n], r) for n, r in DISTRACTORS]
    models = load_models()
    real = load_real(args.real_labels) if args.real_labels else []
    real_train = [it for it in real if it['split'] == 'train']
    real_test = [it for it in real if it['split'] == 'test']
    cuts = real_cutouts(real_train)
    print('real cut-outs per class:', {c: len(v) for c, v in cuts.items()})
    for i, it in enumerate(real_train):     # scope-free areas of the train photos become backgrounds
        photos[f'real{i}'] = it['img']
        OCCUPIED[f'real{i}'] = [(b['bbox'][0], b['bbox'][1], b['bbox'][0] + b['bbox'][2], b['bbox'][1] + b['bbox'][3])
                                for b in it['boxes']]

    scenes = []
    count = {c: 0 for c in CLASSES}
    while min(count.values()) < args.per_class:
        k = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0]
        # favour the classes that are behind
        pool = sorted(CLASSES, key=lambda c: count[c] + random.random() * 20)[:k]
        img, anns = make_scene(list(pool), models, distractors, photos, cuts, args.cutout_frac)
        for cls, _ in anns:
            count[cls] += 1
        scenes.append((img, anns))
    for _ in range(int(len(scenes) * args.neg_frac)):
        scenes.append(make_scene([], models, distractors, photos))
    random.shuffle(scenes)

    # the real train photos themselves, resized like the synthetic images
    for it in real_train:
        img = it['img']
        s = 640 / max(img.shape[:2])
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        scenes.append((small, [(b['cls'], [round(v * s) for v in b['bbox']]) for b in it['boxes']]))
    random.shuffle(scenes)

    if os.path.exists(args.out):
        for split in ('train2017', 'val2017', 'test2017', 'annotations'):
            shutil.rmtree(os.path.join(args.out, split), ignore_errors=True)
    n_val = int(len(scenes) * args.val_frac)
    named = [(img, anns, f'syn3_{i:05d}.jpg') for i, (img, anns) in enumerate(scenes, 1)]
    print('train', *write_split(args.out, 'train', named[n_val:]))
    print('val', *write_split(args.out, 'val', named[:n_val]))
    if real_test:
        test = [(it['img'], [(b['cls'], b['bbox']) for b in it['boxes']], 'real_' + os.path.basename(os.path.dirname(it['file']))
                 + '_' + os.path.basename(it['file'])) for it in real_test]
        print('test (real photos)', *write_split(args.out, 'test', test))

    colours = {'rs_rtb2004': (0, 200, 90), 'tek_tds2014': (0, 140, 255), 'tek_tds1002': (255, 80, 200)}
    tiles = []
    for img, anns, _ in named[n_val:n_val + 24]:
        t = img.copy()
        for cls, b in anns:
            cv2.rectangle(t, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), colours[cls], 2)
            cv2.putText(t, cls, (b[0] + 3, max(14, b[1] - 5)), 0, 0.5, colours[cls], 1, cv2.LINE_AA)
        tiles.append(cv2.resize(t, (320, 240)))
    cv2.imwrite(os.path.join(args.out, 'preview.jpg'), np.vstack([np.hstack(tiles[r * 6:r * 6 + 6]) for r in range(4)]))


if __name__ == '__main__':
    main()
