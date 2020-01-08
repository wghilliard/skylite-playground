import os
from collections import deque
from queue import Queue

import numpy as np
from typing import List, Dict
import cv2
from PIL import Image
import logging as lg

lg.basicConfig(level=lg.INFO)


class ColorNode(object):
    mean: np.array = None
    covariance: np.array = None
    class_id: int = None

    left = None  # ColorNode
    right = None  # ColorNode

    def __init__(self, class_id: int):
        self.class_id = class_id


def get_dominant_palette(colors: List[np.array]) -> np.array:
    tile_size: int = 64
    output: np.array = np.zeros((tile_size, len(colors)))

    for index, color in enumerate(colors):
        cv2.rectangle(output, (index * tile_size), (0, tile_size), color)

    return output


def find_dominant_colors(image: np.ndarray, n_colors: int):
    width: int = image.shape[1]
    height: int = image.shape[0]

    colors = np.ones((height, width), dtype=np.int)

    root_node = ColorNode(1)

    get_class_mean_cov(image, colors, root_node)
    for _ in range(1, n_colors):
        next_node = get_max_eigenvalue_node(root_node)
        partition_color(image, colors, get_next_classid(root_node), next_node)
        get_class_mean_cov(image, colors, next_node.left)
        get_class_mean_cov(image, colors, next_node.right)

    found_colors: List[np.ndarray] = get_dominant_colors(root_node)

    quantized_image = get_quantized_image(colors, root_node)
    # quant = Image.fromarray(quantized_image)
    # quant.save('tmp.jpg', 'JPEG')
    # cv::Mat quantized = get_quantized_image(classes, root);
    # cv::Mat
    # viewable = get_viewable_image(classes);
    # cv::Mat
    # dom = get_dominant_palette(colors);
    #
    # cv::imwrite("./classification.png", viewable);
    # cv::imwrite("./quantized.png", quantized);
    # cv::imwrite("./palette.png", dom);
    #
    # return colors;

    return list(map(lambda color: color.astype(np.int), found_colors)), quantized_image


def get_class_mean_cov(image: np.ndarray, colors: np.ndarray, node: ColorNode):
    width: int = image.shape[1]
    height: int = image.shape[0]
    class_id: int = node.class_id

    color_sum: np.ndarray = np.zeros((3,), dtype=np.float)
    scaled_sum: np.ndarray = np.zeros((3, 3), dtype=np.float)

    count: int = 0

    for y in range(0, height):
        for x in range(0, width):
            if colors[y][x] != class_id:
                continue

            org_color: np.ndarray = image[y][x]
            scaled_color: np.ndarray = org_color / 255.0

            color_sum += scaled_color
            scaled_sum += (scaled_color * scaled_color.transpose())

            count += 1

    covariance = scaled_sum - (color_sum * color_sum.transpose()) / count
    mean = color_sum / count

    node.mean = np.copy(mean)
    node.covariance = np.copy(covariance)


def get_max_eigenvalue_node(node: ColorNode) -> ColorNode:
    if node.left is None and node.right is None:
        return node

    max_eigen_value = -1
    # eigenvalues = np.zer
    queue = deque()
    queue.append(node)

    output = node
    while len(queue) > 0:
        tmp_node: ColorNode = queue.popleft()

        if tmp_node.right and tmp_node.left:
            queue.append(tmp_node.right)
            queue.append(tmp_node.left)
            continue

        retval, eigenvalues, eigenvectors = cv2.eigen(tmp_node.covariance)

        if (eigenvalues[0] > max_eigen_value):
            max_eigen_value = eigenvalues[0]
            output = tmp_node

    return output


def partition_color(image: np.ndarray, colors: np.ndarray, next_class_id: int, node: ColorNode):
    width: int = image.shape[1]
    height: int = image.shape[0]
    class_id: int = node.class_id

    new_left_id: int = next_class_id
    new_right_id: int = next_class_id + 1

    mean: np.ndarray = node.mean
    covariance: np.ndarray = node.covariance
    _, eigenvalues, eigenvectors = cv2.eigen(covariance)

    best_eig_vec: np.ndarray = eigenvectors[0]

    threshold_value = best_eig_vec.dot(mean)
    # threshold_value = best_eig_vec * mean

    node.left = ColorNode(new_left_id)
    node.right = ColorNode(new_right_id)

    for y in range(0, height):
        for x in range(0, width):
            if colors[y][x] != class_id:
                continue

            org_color: np.ndarray = image[y][x]
            scaled_color: np.ndarray = org_color / 255.0

            projected_color: np.ndarray = best_eig_vec.dot(scaled_color)
            # projected_color: np.ndarray = best_eig_vec * scaled_color

            if projected_color <= threshold_value:
                # if np.all(projected_color <= threshold_value):
                colors[y][x] = new_left_id
            else:
                colors[y][x] = new_right_id


def get_next_classid(node: ColorNode) -> int:
    max_id: int = 0
    queue: deque = deque()
    queue.append(node)

    while len(queue) > 0:
        tmp_node: ColorNode = queue.popleft()
        if tmp_node.class_id:
            max_id = max(tmp_node.class_id, max_id)

        # - b/c we always split in two, all non-leaf nodes should have two children
        # if a left exists, so should a right
        # - we'd see a NoneType exception around here if both children were not being
        # initialized / linked properly
        if tmp_node.left:
            queue.append(tmp_node.left)
            queue.append(tmp_node.right)

    return max_id + 1


def get_dominant_colors(node: ColorNode) -> List[np.ndarray]:
    return list(map(lambda tmp_node: tmp_node.mean * 255, get_leaves(node)))


def get_leaves(node: ColorNode) -> List[ColorNode]:
    output = list()
    queue: deque = deque()
    queue.append(node)

    while len(queue) > 0:
        tmp_node = queue.popleft()
        if tmp_node.right:
            queue.append(tmp_node.right)
            queue.append(tmp_node.left)
            continue

        output.append(tmp_node)

    return output


def get_quantized_image(colors: np.ndarray, node: ColorNode) -> np.ndarray:
    leaves_map: Dict[int, np.ndarray] = {n.class_id: n.mean for n in get_leaves(node)}

    width: int = colors.shape[1]
    height: int = colors.shape[0]

    output = np.zeros((height, width, 3), dtype=np.float)

    for y in range(0, height):
        for x in range(0, width):
            color_class_id: int = colors[y][x]
            output[y][x] = np.copy(leaves_map[color_class_id])

    return output


def main(args: dict):
    filename = args.get('filename')
    lg.info(os.stat(filename))
    # image = cv2.imread(filename, 0)
    image = Image.open(filename)

    sampled_image = np.array(image.resize((400, 300)))
    # TODO - wgh
    colors, quantized_image = find_dominant_colors(sampled_image, args.get('n_colors'))
    print(colors)
    # return image
    np.concatenate

if __name__ == '__main__':
    main({'filename': '../data/training/sunset_1.jpg',
          'n_colors': 2})
