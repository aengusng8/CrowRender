import torch
from torch import autocast
from torch.utils.data import Dataset
import json
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import numpy as np
import random
import copy
from einops import repeat

from einops import rearrange
from tqdm import tqdm, trange
from contextlib import contextmanager, nullcontext
from pytorch_lightning import seed_everything


def encode_from_custom_annotation(custom_annotations, size=512):
    #     custom_annotations = [
    #     {'x': 83, 'y': 335, 'width': 70, 'height': 69, 'label': 'blue metal cube'},
    #     {'x': 162, 'y': 302, 'width': 110, 'height': 138, 'label': 'blue metal cube'},
    #     {'x': 274, 'y': 250, 'width': 191, 'height': 234, 'label': 'blue metal cube'},
    #     {'x': 14, 'y': 18, 'width': 155, 'height': 205, 'label': 'blue metal cube'},
    #     {'x': 175, 'y': 79, 'width': 106, 'height': 119, 'label': 'blue metal cube'},
    #     {'x': 288, 'y': 111, 'width': 69, 'height': 63, 'label': 'blue metal cube'}
    # ]
    H, W = size, size

    objects = []
    for j in range(len(custom_annotations)):
        xyxy = [
            custom_annotations[j]["x"],
            custom_annotations[j]["y"],
            custom_annotations[j]["x"] + custom_annotations[j]["width"],
            custom_annotations[j]["y"] + custom_annotations[j]["height"],
        ]
        objects.append(
            {
                "caption": custom_annotations[j]["label"],
                "bbox": xyxy,
            }
        )

    out = encode_scene(
        objects, H=H, W=W, src_bbox_format="xyxy", tgt_bbox_format="xyxy"
    )

    return out
