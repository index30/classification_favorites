import fire
import numba
import numpy as np
from pathlib import Path
from requests_oauthlib import OAuth1Session
import time
import tweepy

import create_data
import get_tweet
import model
import secrets
from operation import basic

class Allocate(object):
    def __init__(self):
        self.oauth = tweepy.OAuthHandler(secrets.CKey, secrets.CSecret)
        self.oauth.set_access_token(secrets.AToken, secrets.ASecret)
        self.twitter = tweepy.API(self.oauth)
        self.genre = ["animal", "illust", "scenery"]

    def register_tweet(self, page_num=0, SAVE_DIR="images/favorite"):
        get_tweet.save_favorites(self.twitter, page_num, SAVE_DIR)
        print("finish regist")

    @numba.jit
    def main(self, SAVE_DIR="images/favorite", ALLOCATE_PATH="images/allocate", MODEL_NAME="use_model/model.h5", IMG_SIZE=256, CLASSES=3):
        basic.mkdir(Path(ALLOCATE_PATH))
        target_list = []
        target_list.extend(Path(SAVE_DIR).glob("*.jpg"))
        target_list.extend(Path(SAVE_DIR).glob("*.png"))
        for target in target_list:
            start = time.time()
            predict = model.predict(target, MODEL_NAME, IMG_SIZE, CLASSES)
            elapsed_time = time.time() - start
            print("time: {}".format(elapsed_time))
            target_genre = self.genre[np.argmax(predict)]
            basic.read_write_img(str(Path(ALLOCATE_PATH, target_genre, target.name)), target)

if __name__=="__main__":
    fire.Fire(Allocate)
