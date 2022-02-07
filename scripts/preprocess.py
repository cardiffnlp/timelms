"""
Expects concatenated API responses output by scripts/combine.py

Requires:
- The file data/verified_users.v310821.txt
$ pip install datasketch==1.5.3
$ pip install xxhash==2.0.2

Usage:
$ python scripts/preprocess.py <input .jl> <output .jl>
$ python scripts/preprocess.py tweets-2020-Q3.jl tweets-2020-Q3.cleaned.jl
"""

import sys
import json
import logging
import string
from collections import Counter

from datasketch import MinHash, LeanMinHash
import xxhash


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


verified_users = set(open("data/verified_users.v310821.txt").read().split('\n'))


def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    new_text = []
    for t in text.split():
        t = '@user' if t.startswith('@') and len(t) > 1 and t.replace('@','') not in verified_users else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    
    return ' '.join(new_text)


def hash_tweet(tweet, num_perm=16):

    def normalize_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        text = text.lower()
        return text

    def minhash(seq):
        # https://skeptric.com/minhash/
        m = MinHash(num_perm=num_perm, hashfunc=xxhash.xxh64_intdigest)
        for s in seq:
            m.update(s.encode('utf8'))
        return LeanMinHash(m)

    tokens = normalize_text(tweet['text']).split()  # whitespace tokenization
    return minhash(tokens)



if __name__ == '__main__':

    input_tweets_fn = sys.argv[1]
    output_tweets_fn = sys.argv[2]

    blacklist_pct = 0.01

    logging.info('1st pass - Collecting username counts ...')
    n_input_tweets = 0
    user_counter = Counter()
    with open(input_tweets_fn) as in_tweets_f:

        for idx, jl_str in enumerate(in_tweets_f):
            if idx % 1e6 == 0:
                logging.info('1st pass - at idx %d' % idx)

            tweet = json.loads(jl_str)

            user_counter[tweet['username']] += 1
            n_input_tweets += 1

    logging.info('1st pass - Completed, found %d tweets' % n_input_tweets)
    logging.info('1st pass - Found %d users' % len(user_counter.keys()))

    blacklisted_users = set()
    top_users = [user for user, _ in user_counter.most_common()]

    n_blacklisted_users = int(len(top_users)*blacklist_pct)
    blacklisted_users = set(top_users[:n_blacklisted_users])
    
    # additional stats
    n_users = len(user_counter.keys())
    pct_blacklisted_users = round((n_blacklisted_users / n_users) * 100, 2)

    n_blacklisted_tweets = sum([user_counter[u] for u in blacklisted_users])
    pct_blacklisted_tweets = round((n_blacklisted_tweets / sum(user_counter.values())) * 100, 2)

    logging.info(f"1st pass - Blacklisted {len(blacklisted_users)} users ({pct_blacklisted_users}%), ignoring {n_blacklisted_tweets} tweets ({pct_blacklisted_tweets}%)")


    logging.info('2nd pass - Hashing and writing valid tweets ...')

    written_hashes = set()
    n_written = 0
    n_ignored_by_user = 0
    n_ignored_by_hash = 0
    with open(input_tweets_fn) as in_tweets_f:

        with open(output_tweets_fn, 'w') as out_tweets_f:

            for idx, jl_str in enumerate(in_tweets_f):
                # if idx % 1e6 == 0:
                if idx % 1e5 == 0:
                    logging.info('2nd pass - at idx %d' % idx)

                tweet = json.loads(jl_str)
                tweet['text'] = clean_text(tweet['text'])

                if tweet['username'] in blacklisted_users:
                    n_ignored_by_user += 1
                    continue

                tweet_hash = hash_tweet(tweet)

                if tweet_hash not in written_hashes:
                    out_tweets_f.write(json.dumps(tweet)+'\n')
                    n_written += 1

                    written_hashes.add(tweet_hash)

                else:
                    n_ignored_by_hash += 1

    logging.info(f"2nd pass - Completed, wrote {n_written} tweets, ignored {n_ignored_by_user} by user blacklist, {n_ignored_by_hash} by hash collision")

    logging.info("Done")
