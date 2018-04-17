from requests_oauthlib import OAuth1Session
import json

### This file has consumer key etc.
import secrets

all_url = "https://api.twitter.com/1.1/statuses/home_timeline.json"
fav_url = "https://api.twitter.com/1.1/favorites/list.json"
params = {}

twitter = OAuth1Session(secrets.CKey, secrets.CSecret, secrets.AToken, secrets.ASecret)
request = twitter.get(fav_url, params=params)

if request.status_code == 200:
    timeline = json.loads(request.text)
    for tweet in timeline:
        print(tweet["text"])
else:
    print("Error:{}".format(request.status_code))
