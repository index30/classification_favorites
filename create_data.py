from pathlib import Path
import random
import skimage.io
import shutil

ROOT_PATH = Path("images")
FOR_TRAIN_PATH = Path(ROOT_PATH, "for_train")
DATA_PATH = Path(ROOT_PATH, "data")

### ジャンル区別用
genre = {'animal':['犬', '猫'], 'scenery':['風景', '神社'], 'illust':['東方', 'FGOイラスト']}


def mkdir(path):
    if not path.exists():
        path.mkdir()


def read_write_img(save_path, img_path):
    skimage.io.imsave(save_path, skimage.io.imread(img_path))


def save_train_test(img_list, parent_dir, g_s="", split_num=3):
    if Path(DATA_PATH, "train", parent_dir).exists():
        shutil.rmtree(Path(DATA_PATH, "train", parent_dir))
        shutil.rmtree(Path(DATA_PATH, "validation", parent_dir))
    mkdir(Path(DATA_PATH, "train", parent_dir))
    mkdir(Path(DATA_PATH, "validation", parent_dir))
    list_len = len(img_list)
    random.shuffle(img_list)
    for (i, img) in enumerate(img_list):
        try:
            if i < list_len*(1/split_num):
                if img.suffix == '.jpg':
                    read_write_img(str(Path(DATA_PATH, "validation", parent_dir, g_s+img.stem+".jpg")), img)
                else:
                    read_write_img(str(Path(DATA_PATH, "validation", parent_dir, g_s+img.stem+".png")), img)
            else:
                if img.suffix == '.jpg':
                    read_write_img(str(Path(DATA_PATH, "train", parent_dir+g_s+img.stem+".jpg")), img)
                else:
                    read_write_img(str(Path(DATA_PATH, "train", parent_dir+g_s+img.stem+".png")), img)
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
    mkdir(DATA_PATH)
    mkdir(Path(DATA_PATH, 'train'))
    mkdir(Path(DATA_PATH, 'validation'))
    main()
