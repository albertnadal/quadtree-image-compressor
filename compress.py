import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab, deltaE_ciede2000
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import math
import sys
import time

COLOR_DELTA = 6.0
MAX_RECUSIVE_CALLS = 100
sys.setrecursionlimit(MAX_RECUSIVE_CALLS)

image = Image.open("original.jpg")
image = image.convert("RGB")
pixels = image.load()

image_tree = image.copy()
image_tree_draw = ImageDraw.Draw(image_tree)

@dataclass
class Rect:
    sx: int
    sy: int
    ex: int
    ey: int

@dataclass
class Color:
    r: float
    g: float
    b: float

class QuadNode:
    def __init__(self, region: Rect, color: Color):
        self.region = region
        self.color = color
        self.first_node = None
        self.second_node = None
        self.third_node = None
        self.forth_node = None

def delta_e(rgb1: Color, rgb2: Color):
    lab1 = rgb2lab(np.array([rgb1.r, rgb1.g, rgb1.b]).reshape(1, 1, 3) / 255)
    lab2 = rgb2lab(np.array([rgb2.r, rgb2.g, rgb2.b]).reshape(1, 1, 3) / 255)
    return float(deltaE_ciede2000(lab1, lab2)[0, 0])

def compress_image(region: Rect) -> Tuple[bool, Color]:
    # Get the average color of the region
    avg_color: Color = Color(0, 0, 0)
    start = time.perf_counter()
    total_pixels: int = 0

    for x in range(region.sx, region.ex):
        for y in range(region.sy, region.ey):
            # TODO: Next lines are O(n). We refactor de the code by avoiding recalculate
            #       pixel sums by using a 2D prefix sum (integral images) with an O(1) cost.
            pixel: Tuple[int, int, int] = pixels[x, y]
            avg_color.r += pixel[0]
            avg_color.g += pixel[1]
            avg_color.b += pixel[2]
            total_pixels += 1
    final = time.perf_counter()
    time_ms = (final - start) * 1000
    print(f"Time: {time_ms:.3f} ms")

    #total_pixels: int = (region.ex - region.sx + 1) * (region.ey - region.sy + 1)
    avg_color.r /= total_pixels
    avg_color.g /= total_pixels
    avg_color.b /= total_pixels
    print(f"TOTAL PIXELS: ({total_pixels}) RGB: ({avg_color.r}, {avg_color.g}, {avg_color.b})")

    if (region.ex - region.sx <= 1) or (region.ey - region.sy <= 1):
        # A leaf node has been reached
        return (False, avg_color)

    need_fragment: bool = False

    first_quad_rect: Rect = Rect(region.sx, region.sy, (region.sx+region.ex)//2, (region.sy+region.ey)//2)
    if (region.sx < first_quad_rect.ex) and (region.sy < first_quad_rect.ey):
        avg_color_first_quad: Color
        fragmented: bool
        (fragmented, avg_color_first_quad) = compress_image(first_quad_rect)
        need_fragment = need_fragment | fragmented
        if delta_e(avg_color_first_quad, avg_color) >= COLOR_DELTA:
            need_fragment = True
            #TODO: store subnode fragmentation codification

    second_quad_rect: Rect = Rect(((region.sx+region.ex)//2) + 1, region.sy, region.ex, (region.sy+region.ey)//2)
    if (second_quad_rect.sx < region.ex) and (region.sy < second_quad_rect.ey):
        avg_color_second_quad: Color
        fragmented: bool
        (fragmented, avg_color_second_quad) = compress_image(second_quad_rect)
        need_fragment = need_fragment | fragmented
        if need_fragment or delta_e(avg_color_second_quad, avg_color) >= COLOR_DELTA:
            need_fragment = True
            #TODO: store subnode fragmentation codification

    third_quad_rect: Rect = Rect((region.sx+region.ex)//2 + 1, (region.sy+region.ey)//2 + 1, region.ex, region.ey)
    if (third_quad_rect.sx < region.ex) and (third_quad_rect.sy < region.ey):
        avg_color_third_quad: Color
        fragmented: bool
        (fragmented, avg_color_third_quad) = compress_image(third_quad_rect)
        need_fragment = need_fragment | fragmented
        if need_fragment or delta_e(avg_color_third_quad, avg_color) >= COLOR_DELTA:
            need_fragment = True
            #TODO: store subnode fragmentation codification

    forth_quad_rect: Rect = Rect(region.sx, (region.sy+region.ey)//2 + 1, (region.sx+region.ex)//2, region.ey)
    if (region.sx < (region.sx+region.ex)//2) and ((region.sy+region.ey)//2 + 1 < region.ey):
        avg_color_forth_quad: Color
        fragmented: bool
        (fragmented, avg_color_forth_quad) = compress_image(forth_quad_rect)
        need_fragment = need_fragment | fragmented
        if need_fragment or delta_e(avg_color_forth_quad, avg_color) >= COLOR_DELTA:
            need_fragment = True
            #TODO: store subnode fragmentation codification

    if need_fragment:
        image_tree_draw.rectangle((first_quad_rect.sx, first_quad_rect.sy, first_quad_rect.ex, first_quad_rect.ey), outline=(0, 0, 0), width=1)
        image_tree_draw.rectangle((second_quad_rect.sx, second_quad_rect.sy, second_quad_rect.ex, second_quad_rect.ey), outline=(0, 0, 0), width=1)
        image_tree_draw.rectangle((third_quad_rect.sx, third_quad_rect.sy, third_quad_rect.ex, third_quad_rect.ey), outline=(0, 0, 0), width=1)
        image_tree_draw.rectangle((forth_quad_rect.sx, forth_quad_rect.sy, forth_quad_rect.ex, forth_quad_rect.ey), outline=(0, 0, 0), width=1)

    return (need_fragment, avg_color)

compress_image(Rect(0, 0, image.width - 1, image.height - 1))
image_tree.save("out.png")