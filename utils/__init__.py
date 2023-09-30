from __future__ import annotations
from itertools import chain
from functools import lru_cache
from pathlib import Path
import random
import re

from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
# import spacy
import torch
import torch.nn.functional as F

def expand_image(im: torch.Tensor, h = 512, w = 512,
                 absolute: bool = False, threshold: float = None) -> torch.Tensor:
    im = im.unsqueeze(0).unsqueeze(0)
    im = F.interpolate(im.float().detach(), size=(h, w), mode='bicubic')
    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
    if threshold:
        im = (im > threshold).float()
    # im = im.cpu().detach()
    return im.squeeze()


def _convert_heat_map_colors(heat_map : torch.Tensor):
    def get_color(value):
        return np.array(cm.turbo(value / 255)[0:3])

    color_map = np.array([ get_color(i) * 255 for i in range(256) ])
    color_map = torch.tensor(color_map, device=heat_map.device)

    heat_map = (heat_map * 255).long()

    return color_map[heat_map]

def image_overlay_heat_map(img,
                           heat_map,
                           word=None, out_file=None, crop=None, alpha=0.5, caption=None, image_scale=1.0):
    # type: (Image.Image | np.ndarray, torch.Tensor, str, Path, int, float, str, float) -> Image.Image
    assert(img is not None)

    if heat_map is not None:
        shape : torch.Size = heat_map.shape
        # heat_map = heat_map.unsqueeze(-1).expand(shape[0], shape[1], 3).clone()
        heat_map = _convert_heat_map_colors(heat_map)
        heat_map = heat_map.to('cpu').detach().numpy().copy().astype(np.uint8)
        heat_map_img = Image.fromarray(heat_map)

        img = Image.blend(img, heat_map_img, alpha)
    else:
        img = img.copy()

    if caption:
        img = _write_on_image(img, caption)

    if image_scale != 1.0:
        x, y = img.size
        size = (int(x * image_scale), int(y * image_scale))
        img = img.resize(size, Image.BICUBIC)
    return img