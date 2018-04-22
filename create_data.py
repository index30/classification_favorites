from pathlib import Path
import random
import skimage.io
import shutil

from operation import basic
import path

ROOT_PATH = path.ROOT_PATH
FOR_TRAIN_PATH = path.FOR_TRAIN_PATH
DATA_PATH = path.DATA_PATH

### ジャンル区別用
genre = {'animal':['犬', '猫'], 'scenery':['風景', '神社'], 'illust':['東方', 'FGOイラスト']}


def save_train_test(img_list, parent_dir, g_s="", split_num=3):
    if Path(DATA_PATH, "train", parent_dir).exists():
        shutil.rmtree(Path(DATA_PATH, "train", parent_dir))
        shutil.rmtree(Path(DATA_PATH, "validation", parent_dir))
    basic.mkdir(Path(DATA_PATH, "train", parent_dir))
    basic.mkdir(Path(DATA_PATH, "validation", parent_dir))
    list_len = len(img_list)
    random.shuffle(img_list)
    for (i, img) in enumerate(img_list):
        try:
            if i < list_len*(1/split_num):
                if img.suffix == '.jpg':
                    basic.read_write_img(str(Path(DATA_PATH, "validation", parent_dir, g_s+img.stem+".jpg")), img)
                else:
                    basic.read_write_img(str(Path(DATA_PATH, "validation", parent_dir, g_s+img.stem+".png")), img)
            else:
                if img.suffix == '.jpg':
                    basic.read_write_img(str(Path(DATA_PATH, "train", parent_dir+g_s+img.stem+".jpg")), img)
                else:
                    basic.read_write_img(str(Path(DATA_PATH, "train", parent_dir+g_s+img.stem+".png")), img)
        except KeyError as e:
            print("{}".format(e))
        except OSError as e:
            print("{}".format(e))


def main():
    for (g, g_short) in genre.items():
        print(g)
        for g_s in g_short:
            print(g_s)
            img_list = []
            path = Path(FOR_TRAIN_PATH, g_s)
            img_list.extend(path.glob("*.jpg"))
            img_list.extend(path.glob("*.png"))
            save_train_test(img_list, g, g_s, 3)


if __name__=="__main__":
    basic.mkdir(DATA_PATH)
    basic.mkdir(Path(DATA_PATH, 'train'))
    basic.mkdir(Path(DATA_PATH, 'validation'))
    main()
