# Classification favorites
## What is it?
For classifying favorites in twitter.

## How use this?
### When get your favorite tweet

1. You must access twitter API and get consumer key etc.
2. You write secrets.py like below.

```python:secrets.py
### consumer key
CKey = '***********'
### consumer secret
CSecret = '**********'
### access token
AToken = '**********'
### access secret
ASecret = '*********'
### owner id
OId = '***********'
```

3. Run get_tweet.py.

### When you prepare data for train

1. You must get image by yourself, and make directory.

```
- images
    - for_train
        - A
        - B
        - ...
```

2. Run create_data.py

### When you train model
Run model.py. You can use below model.

- lenet
- vgg
- xception
- mobilenet(under examination)

### When you allocate your favorite tweet
Run allocate\_tweet\_media.py
