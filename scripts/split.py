"""
Should be used with tweets (.jl) preprocessed by scripts/preprocess.py
"""

import sys
import json
import logging
from collections import Counter, defaultdict

import random
random.seed(42)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


if __name__ == '__main__':

    # TODO: argparse
    input_path = sys.argv[1]  # preprocessed .jl
    train_path = sys.argv[2]  # output path for training tweets (.jl)
    val_path = sys.argv[3]  # output path for validation tweets (.jl)

    n_train_monthly = 1400000
    n_validation_monthly = 50000


    # load
    logging.info("Loading tweets ...")
    tweets = defaultdict(list)
    n_failed = 0
    dist_input = Counter()
    with open(input_path) as jl_f:
        for jl_idx, jl_str in enumerate(jl_f):
            if jl_idx % 1e6 == 0:
                logging.info(f"at {jl_idx}, {n_failed} failed")

            try:
                jl = json.loads(jl_str)
            except json.decoder.JSONDecodeError:
                logging.warn('Failed to process line %d:')
                logging.warn('%s\n' % jl_str)
                n_failed += 1
                continue

            relevant_fields = ['id', 'created_at', 'text', 'username']
            tweet = {k: jl[k] for k in relevant_fields}

            ym = tweet['created_at'][:7]
            dist_input[ym] += 1

            tweets[ym].append(tweet)

    if n_failed > 0:
        logging.warn(f"Failed to process {n_failed} tweets")

    assert len(tweets.keys()) == 3  # 3 YMs (for a quarter)


    # split
    logging.info("Preparing train/val splits ...")

    val_tweets = []
    for ym, ym_tweets in tweets.items():
        val_tweets += random.sample(ym_tweets, n_validation_monthly)
    val_ids = {tw['id'] for tw in val_tweets}

    train_tweets = []
    for ym, ym_tweets in tweets.items():
        ym_tweets = [tw for tw in ym_tweets if tw['id'] not in val_ids]
        train_tweets += random.sample(ym_tweets, n_train_monthly)
    train_ids = {tw['id'] for tw in train_tweets}
    
    val_tweets = sorted(val_tweets, key=lambda x: x['created_at'])
    train_tweets = sorted(train_tweets, key=lambda x: x['created_at'])

    assert len(train_ids.intersection(val_ids)) == 0  # sanity check


    # check
    logging.info('Processing Year-Month distributions ...')
    dist_train = Counter()
    for tweet in train_tweets:
        ym = tweet['created_at'][:7]
        dist_train[ym] += 1

    dist_val = Counter()
    for tweet in val_tweets:
        ym = tweet['created_at'][:7]
        dist_val[ym] += 1

    for ym in sorted(dist_train.keys()):
        print(ym, dist_train[ym], dist_val[ym])

    logging.info(f"{len(train_tweets)} train tweets")
    logging.info(f"{len(val_tweets)} validation tweets")


    # write
    logging.info(f"Writing {train_path} ...")
    with open(train_path, 'w') as train_f:
        for tweet in train_tweets:
            train_f.write('%s\n' % json.dumps(tweet))

    logging.info(f"Writing {val_path} ...")
    with open(val_path, 'w') as val_f:
        for tweet in val_tweets:
            val_f.write('%s\n' % json.dumps(tweet))

    # additionally, writing just the text in .txt format (one tweet per line)
    train_path = train_path.replace('.jl', '.txt')
    val_path = val_path.replace('.jl', '.txt')

    logging.info(f"Writing {train_path} ...")
    with open(train_path, 'w') as train_f:
        for tweet in train_tweets:
            train_f.write('%s\n' % tweet['text'])

    logging.info(f"Writing {val_path} ...")
    with open(val_path, 'w') as val_f:
        for tweet in val_tweets:
            val_f.write('%s\n' % tweet['text'])

    logging.info("Done")
