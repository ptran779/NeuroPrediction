# Utilities func
import numpy as np


def one_hot_convert(raw):
    """
    convert a 1D array to one-hot-coding
    :param raw: an array
    :return: 2D matrix
    """
    cls = np.unique(raw)
    out = np.zeros([len(raw), len(cls)])
    cls_look_up = {item: index for index, item in enumerate(cls)}   # convert array to dict
    for c, i in enumerate(raw):
        out[c, cls_look_up[i]] = 1                                  # mark target code to 1
    return out


def split_data(arr_len, ratio, seed=-1):
    """
    Split data to test and train with numpy random seeding.
    :param arr_len: length of array need split
    :param ratio: 0-1 of item to split, round down to first array
    :param seed: use to control random state -- set to -1 to disable
    :return: tuple of 2 arrays contain index that has been split
    """
    # set random state if used
    if seed != -1:
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(arr_len)
    else:
        shuffled_indices = np.random.permutation(arr_len)
    # split
    split_index = int(arr_len * ratio)
    return shuffled_indices[:split_index], shuffled_indices[split_index:]
