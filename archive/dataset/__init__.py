import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def read_fns(filename):
    with open(filename) as f:
        fns = f.readlines()
        fns = [fn.strip() for fn in fns]
    return fns


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def read_paired_fns(filename):
    paired = []
    with open(filename) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            # only keep the noisy/clean paths; ignore optional metadata columns
            paired.append((tokens[0], tokens[1]))
    return paired
