from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats.mstats import gmean
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .common import get_palette


# Thanks Claudia!
def f0(array):
    return .2 * (255 + 255 + 255 - np.sum(array)) / 3


def f1(array):
    return array.max() - array.min()


def f2(array):
    return 0 if f0(array) >= f1(array) else 1


def get_colors(image: np.ndarray, n_colors: int, write_pca: bool = False):
    pca = PCA(n_components=3)
    print(image.shape)
    X = np.array(image).reshape(-1, 3)
    pca.fit(X)
    samples = np.random.randint(-1000, 2, size=X.shape[0])
    index = np.where(samples > 0, np.ones(shape=X.shape[0]), np.zeros(shape=X.shape[0])).astype(np.int).nonzero()

    Y = pca.transform(X[index]).astype(np.int)

    X_pca_0 = Y[:, 0]
    X_pca_1 = Y[:, 2]

    #     good_colors = np.where(X_pca_0 > 150)[0]
    more_good_colors = np.apply_along_axis(f2, 1, X[index]).nonzero()

    # plot samples in eigenspace
    fig = plt.figure(figsize=(12.8, 9.6))
    ax = fig.add_subplot(111)
    ax.scatter(X_pca_0, X_pca_1, c=X[index] / 255)
    if write_pca:
        ax.figure.savefig('./kmeans-pca_output.png', format='png')
    # plt.show()

    cluster = KMeans(n_colors)
    cluster.fit(X[index][more_good_colors])

    clustered_colors = cluster.predict(X[index][more_good_colors])
    print(clustered_colors.shape)
    color_map = dict()
    for label, color in zip(clustered_colors, X[index][more_good_colors]):
        try:
            color_map[label].append(color)
        except KeyError:
            color_map[label] = [color]

    print('cluster geometric means')
    for label, members in color_map.items():
        value = gmean(np.array(members))

        print(label, len(members), value)
        color_map[label] = value

    return color_map.values()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('image_path', type=str)
    parser.add_argument('-n', '--n_colors', type=int, default=6)
    parser.add_argument('-o', '--output', type=str, default='./')
    parser.add_argument('-w', '--write', action='store_true', default=False)
    args = parser.parse_args()

    n_colors = args.n_colors
    path = Path(args.image_path)
    image_name = path.name.split('.')[0]
    ax = plt.imshow(get_palette(get_colors(np.array(Image.open(str(path)).convert('RGB')), n_colors, write_pca=args.write)))
    ax.figure.savefig(Path(args.output).joinpath(f'kmeans_output_{image_name}_{n_colors}.png'), format='png')
