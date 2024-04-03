from PIL import Image
from tqdm import tqdm
import typing as tp
from dataclasses import dataclass
from collections import defaultdict
import random
import numpy as np


def norm_hamming(h1, h2):
    return (h1 - h2) / len(h1)


def compute_hashes(images, hash_func,
                   distortion_func: tp.Callable = lambda x: x,
                   limit: int | None = None):
    hashes = []
    img: Image.Image
    for idx, img in tqdm(enumerate(images)):
        if limit and idx >= limit:
            break
        img_hash = hash_func(distortion_func(img))
        hashes.append(img_hash)
    return hashes


def avg(x):
    return sum(x) / len(x)


def median(x):
    return sorted(x)[len(x) // 2]


def get_dists_stats(dists):
    return min(dists), max(dists), avg(dists)  # , median(dists)


@dataclass
class S:
    hash_name: str
    orig_hashes: tp.List
    param2dist: tp.Dict[float, tp.List[float]]
    param2distorted_hashes: tp.Dict[float, list]
    limit: int | None = None

    def stats(self):
        return {param: get_dists_stats(dists) for param, dists in self.param2dist.items()}


def compute_distortions_dists(images, distortion_funcs: dict[float, tp.Callable], hash_func,
                              orig_hashes: list | None = None, limit: int | None = None) -> S:
    s = S(hash_name=hash_func.__name__,
          orig_hashes=orig_hashes if orig_hashes else compute_hashes(images, hash_func, limit=limit),
          param2dist=defaultdict(lambda: []),
          param2distorted_hashes={},
          limit=limit)
    for param, distortion_func in distortion_funcs.items():
        distorted_hashes = compute_hashes(images, hash_func, distortion_func, limit)
        s.param2distorted_hashes[param] = distorted_hashes
        for idx in range(len(distorted_hashes)):
            s.param2dist[param].append(norm_hamming(s.orig_hashes[idx], distorted_hashes[idx]))
    return s


def compute_distortions_dists_preloaded(images, distortion_res, hash_func):
    s = S(hash_name=hash_func.__name__,
          orig_hashes=compute_hashes(images, hash_func),
          param2dist=defaultdict(lambda: []),
          param2distorted_hashes={},
          limit=len(images))
    for param, distorted_images in distortion_res.items():
        distorted_hashes = compute_hashes(distorted_images, hash_func)
        s.param2distorted_hashes[param] = distorted_hashes
        for idx in range(len(distorted_hashes)):
            s.param2dist[param].append(norm_hamming(s.orig_hashes[idx], distorted_hashes[idx]))
    return s


def compute_random_dists(orig_hashes, num: int = 100):
    hashes = random.sample(orig_hashes, num)
    other_hashes = random.sample(orig_hashes, num)

    dists = []
    for h in hashes:
        for q in other_hashes:
            dists.append(norm_hamming(h, q))
    random_dists = np.array(dists)
    return random_dists


def get_params(res, hash_name='phash'):
    return res[hash_name].param2dist.keys()
