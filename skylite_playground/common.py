import numpy as np


def get_palette(color_values):
    palette = np.zeros((50, 50 * len(color_values), 3), dtype=np.int)
    for index, color in enumerate(color_values):
        palette[:, index * 50: (index + 1) * 50] = color
    return palette
