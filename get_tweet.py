import json
from pathlib import Path
from requests_oauthlib import OAuth1Session
import tweepy
import urllib.request
import urllib.error

### This file has consumer key etc.
import secrets

SAVE_DIR = "images/favorite"


def save_imgs(entities, save_dir):
    if 'media' in entities:
        for entity in entities['media']:
            img_url = entity['media_url']
            img_name = img_url.split("/")[-1]
            save_Path = Path(save_dir, img_name)
            try:
                response = urllib.request.urlopen(img_url).read()
                with open(save_Path, "wb") as f:
                    f.write(response)
                print("Success saved")
            except urllib.error.URLError as e:
                print("{}".format(e))


def get_favorites(twitter, page_num):
    ### 20tweetだけ取得する
    ### 一定の時間内で使用回数が制限
    #home_tweets = twitter.home_timeline()
    #print(home_tweets)

    ### favのツイートを遡れる
    ### パラメータのpageをいじれば5ヶ月分くらい??
    favorites = twitter.favorites(page=page_num)
    return favorites


def save_favorites(twitter, page_num, save_dir=SAVE_DIR):
    favorites = get_favorites(twitter, page_num)
    for favorite in favorites:
        try:
            save_imgs(favorite.entities, save_dir)
            ext_entities = favorite.extended_entities
            save_imgs(ext_entities, save_dir)
        except TypeError as e:
            print("{}".format(e))
        except AttributeError as e:
            print("{}".format(e))


if __name__=="__main__":
    oauth = tweepy.OAuthHandler(secrets.CKey, secrets.CSecret)
    oauth.set_access_token(secrets.AToken, secrets.ASecret)
    twitter = tweepy.API(oauth)
    get_favorites(twitter, 0)
