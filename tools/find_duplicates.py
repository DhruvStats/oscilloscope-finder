"""List photos that appear in more than one raw folder (same image, maybe re-encoded), to avoid train/test leakage."""
import glob
import os

import cv2
import numpy as np


def sig(p):
    im = cv2.imread(p, 0)
    return cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32), im.shape


folders = ['raw/photos', 'raw/rtb_session', 'raw/photos_tds1002']
sigs = {p: sig(p) for f in folders for p in sorted(glob.glob(f + '/*.jpg'))}
paths = list(sigs)
for i, p in enumerate(paths):
    for q in paths[i + 1:]:
        if os.path.dirname(p) == os.path.dirname(q) or sigs[p][1] != sigs[q][1]:
            continue
        d = np.abs(sigs[p][0] - sigs[q][0]).mean()
        if d < 3:
            print(os.path.relpath(p).replace(os.sep, '/'), '==', os.path.relpath(q).replace(os.sep, '/'), round(float(d), 1))
