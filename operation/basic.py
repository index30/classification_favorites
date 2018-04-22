from pathlib import Path
import skimage.io

def mkdir(path):
    if not path.exists():
        path.mkdir()


def read_write_img(save_path, img_path):
    skimage.io.imsave(save_path, skimage.io.imread(img_path))
