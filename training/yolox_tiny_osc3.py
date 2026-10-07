"""YOLOX-Tiny experiment: three oscilloscopes (rs_rtb2004, tek_tds2014, tek_tds1002)."""
import os

from yolox.exp import Exp as BaseExp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        # YOLOX-Tiny
        self.depth = 0.33
        self.width = 0.375
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.mosaic_scale = (0.5, 1.5)
        self.enable_mixup = False
        self.multiscale_range = 0

        self.num_classes = 3
        self.data_dir = os.path.join(ROOT, "datasets", "oscilloscopes3")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"

        self.data_num_workers = 0
        self.max_epoch = 40
        self.no_aug_epochs = 6
        self.warmup_epochs = 3
        self.eval_interval = 5
        self.exp_name = "yolox_tiny_osc3"
        self.output_dir = os.path.join(ROOT, "models", "training")
