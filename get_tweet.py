import json
from pathlib import Path
from requests_oauthlib import OAuth1Session
import tweepy
import urllib.request
import urllib.error

### This file has consumer key etc.
import secrets

SAVE_DIR = "images"

oauth = tweepy.OAuthHandler(secrets.CKey, secrets.CSecret)
oauth.set_access_token(secrets.AToken, secrets.ASecret)
twitter = tweepy.API(oauth)

### 20tweetだけ取得する
### 時一定の時間内で使用回数が制限
#home_tweets = twitter.home_timeline()
### favのツイートを遡れる
### パラメータのpageをいじれば5ヶ月分くらい??
favorites = twitter.favorites(page=161)

#for tweet in home_tweets:
#    print(tweet.text)

for favorite in favorites:
    #print(favorite.created_at)
    #print("name:{0}\ncontent:{1}\n".format(favorite.user.name, favorite.text))
    parse_text = favorite.text.split()
    img_url = favorite.entities['media'][0]['media_url']
    #print("img_url is {}".format(img_url))
    img_name = img_url.split("/")[-1]
    #print("img_name is {}".format(img_name))

    ### When you want to get image with origin size
    #img_url = "{}:orig".format(img_url)

    save_Path = Path(SAVE_DIR, img_name)

    ### image download
    try:
        response = urllib.request.urlopen(img_url).read()
        with open(save_Path, "wb") as f:
            f.write(response)
        print("Success saved")
    except urllib.error.URLError as e:
        print("{} Error".format(e))

