import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab, deltaE_ciede2000
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple
from typing import BinaryIO

import math
import sys
import time
import struct

COLOR_DELTA = 6.0
MAX_RECUSIVE_CALLS = 100
sys.setrecursionlimit(MAX_RECUSIVE_CALLS)

source_image = Image.open("original.bmp")
source_image = source_image.convert("RGB")
source_pixels = source_image.load()

#image_tree = source_image.copy()
#image_tree_draw = ImageDraw.Draw(image_tree)

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

    def isLeaf(self) -> bool:
        return self.first_node is None and self.second_node is None and self.third_node is None and self.forth_node is None

def delta_e(rgb1: Color, rgb2: Color):
    lab1 = rgb2lab(np.array([rgb1.r, rgb1.g, rgb1.b]).reshape(1, 1, 3) / 255)
    lab2 = rgb2lab(np.array([rgb2.r, rgb2.g, rgb2.b]).reshape(1, 1, 3) / 255)
    return float(deltaE_ciede2000(lab1, lab2)[0, 0])

def compress_image(region: Rect) -> QuadNode:
    # Get the average color of the region
    avg_color: Color = Color(0, 0, 0)
    start = time.perf_counter()

    for x in range(region.sx, region.ex):
        for y in range(region.sy, region.ey):
            # TODO: Next lines are O(n). We refactor de the code by avoiding recalculate
            #       pixel sums by using a 2D prefix sum (integral images) with an O(1) cost.
            pixel: Tuple[int, int, int] = source_pixels[x, y]
            avg_color.r += pixel[0]
            avg_color.g += pixel[1]
            avg_color.b += pixel[2]
    final = time.perf_counter()
    time_ms = (final - start) * 1000
    #print(f"Avg color calc time: {time_ms:.3f} ms")

    total_pixels: int = (region.ex - region.sx) * (region.ey - region.sy)
    avg_color.r /= total_pixels
    avg_color.g /= total_pixels
    avg_color.b /= total_pixels

    node: QuadNode = QuadNode(region, avg_color)

    if (region.ex - region.sx <= 1) or (region.ey - region.sy <= 1):
        # A leaf node has been reached
        return node

    need_fragment: bool = False

    first_quad_rect: Rect = Rect(region.sx, region.sy, (region.sx+region.ex)//2, (region.sy+region.ey)//2)
    if (region.sx < first_quad_rect.ex) and (region.sy < first_quad_rect.ey):
        first_sub_node: QuadNode = compress_image(first_quad_rect)
        sub_node = first_sub_node
        need_fragment = need_fragment or not sub_node.isLeaf()
        if need_fragment or delta_e(sub_node.color, avg_color) >= COLOR_DELTA:
            node.first_node = sub_node
            need_fragment = True

    second_quad_rect: Rect = Rect((region.sx+region.ex)//2, region.sy, region.ex, (region.sy+region.ey)//2)
    if (second_quad_rect.sx < region.ex) and (region.sy < second_quad_rect.ey):
        second_sub_node: QuadNode = compress_image(second_quad_rect)
        sub_node = second_sub_node
        need_fragment = need_fragment or not sub_node.isLeaf()
        if need_fragment or delta_e(sub_node.color, avg_color) >= COLOR_DELTA:
            node.second_node = sub_node
            need_fragment = True

    third_quad_rect: Rect = Rect((region.sx+region.ex)//2, (region.sy+region.ey)//2, region.ex, region.ey)
    if (third_quad_rect.sx < region.ex) and (third_quad_rect.sy < region.ey):
        third_sub_node: QuadNode = compress_image(third_quad_rect)
        sub_node = third_sub_node
        need_fragment = need_fragment or not sub_node.isLeaf()
        if need_fragment or delta_e(sub_node.color, avg_color) >= COLOR_DELTA:
            node.third_node = sub_node
            need_fragment = True

    forth_quad_rect: Rect = Rect(region.sx, (region.sy+region.ey)//2, (region.sx+region.ex)//2, region.ey)
    if (region.sx < (region.sx+region.ex)//2) and ((region.sy+region.ey)//2 < region.ey):
        forth_sub_node: QuadNode = compress_image(forth_quad_rect)
        sub_node = forth_sub_node
        need_fragment = need_fragment or not sub_node.isLeaf()
        if need_fragment or delta_e(sub_node.color, avg_color) >= COLOR_DELTA:
            node.forth_node = sub_node
            need_fragment = True

    if need_fragment:
        node.first_node = first_sub_node
        node.second_node = second_sub_node
        node.third_node = third_sub_node
        node.forth_node = forth_sub_node

    """
    if need_fragment:
        image_tree_draw.rectangle((first_quad_rect.sx, first_quad_rect.sy, first_quad_rect.ex, first_quad_rect.ey), outline=(0, 0, 0), width=1)
        image_tree_draw.rectangle((second_quad_rect.sx, second_quad_rect.sy, second_quad_rect.ex, second_quad_rect.ey), outline=(0, 0, 0), width=1)
        image_tree_draw.rectangle((third_quad_rect.sx, third_quad_rect.sy, third_quad_rect.ex, third_quad_rect.ey), outline=(0, 0, 0), width=1)
        image_tree_draw.rectangle((forth_quad_rect.sx, forth_quad_rect.sy, forth_quad_rect.ex, forth_quad_rect.ey), outline=(0, 0, 0), width=1)
    """
    return node

def render_node_to_image(node: QuadNode, image: ImageDraw.ImageDraw):
    if node.isLeaf():
        image.rectangle((node.region.sx, node.region.sy, node.region.ex - 1, node.region.ey - 1), fill=(int(node.color.r), int(node.color.g), int(node.color.b)))
        #image.rectangle((node.region.sx, node.region.sy, node.region.ex - 1, node.region.ey - 1), outline=(0, 0, 0), width=1)
        return

    if node.first_node is not None:
        render_node_to_image(node.first_node, image)

    if node.second_node is not None:
        render_node_to_image(node.second_node, image)

    if node.third_node is not None:
        render_node_to_image(node.third_node, image)

    if node.forth_node is not None:
        render_node_to_image(node.forth_node, image)

def render_quadtree_to_image_file(node: QuadNode, width: int, height: int, filename: str):
    image = Image.new("RGB", (width, height), "white")
    image_draw = ImageDraw.Draw(image)
    render_node_to_image(node, image_draw)
    image.save(filename)

root_node: QuadNode = compress_image(Rect(0, 0, source_image.width, source_image.height))
render_quadtree_to_image_file(root_node, source_image.width, source_image.height, "out.png")

def save_quadtree(node: QuadNode, filename: str):
    with open(filename, "wb") as file:
        file.write(struct.pack("<HH", node.region.ex - node.region.sx, node.region.ey - node.region.sy)) # Save image width and height using two 16 bytes unsigned integers

        def save_node(node: QuadNode, file: BinaryIO):
            file.write(struct.pack("<BBB", int(node.color.r), int(node.color.g), int(node.color.b))) # Save the node color using 3 bytes
            sub_nodes_flags: int = 0
            if node.first_node is not None:
                sub_nodes_flags |= 1 << 3

            if node.second_node is not None:
                sub_nodes_flags |= 1 << 2

            if node.third_node is not None:
                sub_nodes_flags |= 1 << 1

            if node.forth_node is not None:
                sub_nodes_flags |= 1 << 0

            file.write(struct.pack("<B", sub_nodes_flags)) # Use 1 byte to save the subnode flags to indicate wich subnodes are stored 

            if node.first_node is not None:
                save_node(node.first_node, file)

            if node.second_node is not None:
                save_node(node.second_node, file)

            if node.third_node is not None:
                save_node(node.third_node, file)

            if node.forth_node is not None:
                save_node(node.forth_node, file)

        save_node(node, file)

save_quadtree(root_node, "compressed.dat")

def load_quadtree(filename: str) -> QuadNode:
    with open(filename, "rb") as file:
        width, height = struct.unpack("<HH", file.read(4))

        def load_node(file: BinaryIO, region: Rect) -> QuadNode:
            r, g, b = struct.unpack("<BBB", file.read(3))
            color: Color = Color(r, g, b)
            node: QuadNode = QuadNode(region, color)
            sub_nodes_flags: int = struct.unpack("<B", file.read(1))[0]

            mid_x: int = (region.sx + region.ex) // 2
            mid_y: int = (region.sy + region.ey) // 2

            if sub_nodes_flags & (1 << 3):
                first_quad_rect: Rect = Rect(region.sx, region.sy, mid_x, mid_y)
                node.first_node = load_node(file, first_quad_rect)

            if sub_nodes_flags & (1 << 2):
                second_quad_rect: Rect = Rect(mid_x, region.sy, region.ex, mid_y)
                node.second_node = load_node(file, second_quad_rect)

            if sub_nodes_flags & (1 << 1):
                third_quad_rect: Rect = Rect(mid_x, mid_y, region.ex, region.ey)
                node.third_node = load_node(file, third_quad_rect)

            if sub_nodes_flags & (1 << 0):
                forth_quad_rect: Rect = Rect(region.sx, mid_y, mid_x, region.ey)
                node.forth_node = load_node(file, forth_quad_rect)

            return node

        root_region: Rect = Rect(0, 0, width, height)
        return load_node(file, root_region)

loaded_node: QuadNode = load_quadtree("compressed.dat")
