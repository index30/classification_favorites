import json
from requests_oauthlib import OAuth1Session
import tweepy

### This file has consumer key etc.
import secrets

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
    print(favorite.created_at)
    print("name:{0}\ncontent:{1}\n".format(favorite.user.name, favorite.text))
