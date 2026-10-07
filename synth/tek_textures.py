"""Build box textures for the Tektronix TDS 2014 / TDS 1002 from the lab phone photos.

Both models share the same TDS1000/2000 case (326 x 158 x 124 mm); only the front differs.
Photo numbers refer to raw/photos/NN.jpg; a "1002/NN" entry refers to raw/photos_tds1002/NN.jpg.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PHOTOS = os.path.join(ROOT, 'raw', 'photos')
OUT = os.path.join(HERE, 'textures')

FW = 1000
FH = int(FW * 158 / 326)
DEPTH = int(FW * 124 / 326)

# photo, quad (tl, tr, br, bl as mapped onto the face), output size
QUADS = {
    'tds2014_front': (34, [(250, 95), (1340, 90), (1336, 585), (256, 547)], (FW, FH)),
    'tds1002_front': (48, [(155, 92), (1328, 106), (1326, 598), (160, 592)], (FW, FH)),
    'back': (56, [(285, 15), (1288, 20), (1288, 548), (285, 548)], (FW, FH)),
    'left': (53, [(105, 388), (408, 352), (488, 1062), (105, 1060)], (DEPTH, FH)),
    'top': (51, [(150, 140), (1445, 140), (1445, 555), (150, 555)], (FW, DEPTH)),
    'bottom': (42, [(152, 92), (1392, 185), (1310, 572), (225, 548)], (FW, DEPTH)),
    # the TDS 1002 has its own back (RS232/Centronics/GPIB module) - photo held sideways, quad rotates it upright
    'tds1002_back': ('1002/07', [(75, 1312), (75, 378), (548, 378), (548, 1312)], (FW, FH)),
    'tds1002_top': ('1002/05', [(132, 282), (1395, 282), (1395, 700), (132, 700)], (FW, DEPTH)),
}


def grabcut_mask(img, quad):
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    x0, y0 = max(0, min(xs) - 4), max(0, min(ys) - 4)
    x1, y1 = min(img.shape[1] - 1, max(xs) + 4), min(img.shape[0] - 1, max(ys) + 4)
    mask = np.zeros(img.shape[:2], np.uint8)
    bg, fg = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, mask, (x0, y0, x1 - x0, y1 - y0), bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    m = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    if cs:
        cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 255, -1)
    return out


def solidify(t, a):
    hole = (a < 140).astype(np.uint8)
    small = cv2.resize(t, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    hs = cv2.dilate(cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST),
                    np.ones((3, 3), np.uint8))
    filled = cv2.GaussianBlur(cv2.inpaint(small, hs * 255, 12, cv2.INPAINT_TELEA), (0, 0), 3)
    filled = cv2.resize(filled, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_CUBIC)
    w = cv2.GaussianBlur(a.astype(np.float32) / 255, (0, 0), 2)[..., None]
    return (t * w + filled * (1 - w)).astype(np.uint8)


def rounded_alpha(w, h, r):
    a = np.zeros((h, w), np.uint8)
    cv2.rectangle(a, (r, 0), (w - 1 - r, h - 1), 255, -1)
    cv2.rectangle(a, (0, r), (w - 1, h - 1 - r), 255, -1)
    for c in [(r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)]:
        cv2.circle(a, c, r, 255, -1, cv2.LINE_AA)
    return a


def lab_match(img, alpha, ref, ref_alpha):
    a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    m, rm = alpha > 128, ref_alpha > 128
    for c in range(3):
        mu, sd = a[..., c][m].mean(), a[..., c][m].std() + 1e-6
        rmu, rsd = r[..., c][rm].mean(), r[..., c][rm].std()
        a[..., c] = (a[..., c] - mu) * (rsd / sd) + rmu
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def build():
    os.makedirs(OUT, exist_ok=True)
    tex = {}
    for name, (n, quad, (w, h)) in QUADS.items():
        if isinstance(n, str):
            img = cv2.imread(os.path.join(ROOT, 'raw', 'photos_tds1002', n.split('/')[1] + '.jpg'))
        else:
            img = cv2.imread(os.path.join(PHOTOS, f'{n:02d}.jpg'))
        mask = grabcut_mask(img, quad)
        M = cv2.getPerspectiveTransform(np.float32(quad), np.float32([(0, 0), (w, 0), (w, h), (0, h)]))
        t = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpPerspective(mask, M, (w, h))
        tex[name] = (t, a)
    # the side, top and bottom shots were taken in different light: match the teal to the back panel
    ref_t, ref_a = tex['back']
    for name in ('left', 'top', 'bottom', 'tds1002_back', 'tds1002_top'):
        t, a = tex[name]
        tex[name] = (lab_match(t, a, ref_t, ref_a), a)
    for name, (t, a) in tex.items():
        t = solidify(t, a)
        cv2.imwrite(os.path.join(OUT, f'tek_{name}.png'), np.dstack([t, rounded_alpha(t.shape[1], t.shape[0], 12)]))
    left = cv2.imread(os.path.join(OUT, 'tek_left.png'), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(OUT, 'tek_right.png'), cv2.flip(left, 1))   # the case is symmetric


if __name__ == '__main__':
    build()
    names = ['tds2014_front', 'tds1002_front', 'back', 'tds1002_back', 'left', 'top', 'tds1002_top', 'bottom']
    tiles = []
    for n in names:
        t = cv2.imread(os.path.join(OUT, f'tek_{n}.png'))
        s = 200 / t.shape[0]
        tiles.append(cv2.resize(t, (int(t.shape[1] * s), 200)))
    rows = [np.hstack(tiles[:2]), np.hstack(tiles[2:5]), np.hstack(tiles[5:7]), np.hstack(tiles[7:])]
    W = max(r.shape[1] for r in rows)
    sheet = np.vstack([np.hstack([r, np.full((200, W - r.shape[1], 3), 255, np.uint8)]) for r in rows])
    cv2.imwrite(os.path.join(OUT, 'tek_net.jpg'), sheet)
