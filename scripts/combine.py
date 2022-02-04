"""
Combines API responses for a given set of YYYY-MMs into a single file.
Expects folders with API responses retrieved using scripts/sampler_api.py

Usage:
$ python scripts/combine.py <output_file> <months:YYYY-MM>
$ python scripts/combine.py tweets-2020-Q3.jl 2020-01 2020-02 2020-03
"""

import sys
import json
import logging
from collections import Counter
from os import listdir


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


if __name__ == '__main__':

    merged_fn = sys.argv[1]
    target_yms = set(sys.argv[2:])


    # collect all response paths
    directories = []
    directories.append('data/responses/')
    directories.append('/media/dan/M2/meaning-shift-data/twitter/00m/')

    all_response_fns = []
    for dir in directories:
        for fn in listdir(dir):

            start_time, end_time = fn.split('.')[-3].split('_')

            end_ym = end_time[:7]
            if len(target_yms) > 0 and end_ym not in target_yms:
                continue

            if fn.endswith('.response.json'):
                all_response_fns.append(dir+fn)

    all_response_fns = sorted(all_response_fns, key=lambda x: x.split('/')[-1])


    n_tweets = 0
    n_failed_location = 0
    n_duplicate_ids = 0
    username_counter = Counter()

    seen_ids = set()
    with open(merged_fn, 'w') as merged_f:
        for fn_idx, fn in enumerate(all_response_fns):

            logging.info('Processing %d/%d - %s' % (fn_idx, len(all_response_fns), fn))
            logging.info('n_tweets: %d' % n_tweets)
            logging.info('n_usernames: %d' % len(username_counter))
            logging.info('n_failed_location: %d' % n_failed_location)
            logging.info('n_duplicate_ids: %d\n' % n_duplicate_ids)

            doc_tweets = []
            with open(fn) as f:
                json_data = json.load(f)

                response_fields = set(json_data['response']['includes'].keys())

                # collect location info
                places_info = {}
                if 'places' in response_fields:
                    for place_entry in json_data['response']['includes']['places']:
                        places_info[place_entry['id']] = {}
                        places_info[place_entry['id']]['place_country'] = place_entry.get('country', '')
                        places_info[place_entry['id']]['place_name'] = place_entry.get('name', '')
                        places_info[place_entry['id']]['place_full_name'] = place_entry.get('full_name', '')

                users_info = {}
                if 'users' in response_fields:
                    for user_entry in json_data['response']['includes']['users']:
                        users_info[user_entry['id']] = {}
                        users_info[user_entry['id']]['user_location'] = user_entry.get('location', '')
                        users_info[user_entry['id']]['username'] = user_entry.get('username', '')

                for tweet in json_data['response']['data']:

                    # merge location info
                    tweet['location'] = {'place_country': '', 'place_name': '', 'place_full_name': '', 'user_location': ''}

                    try:
                        if 'geo' in tweet and 'place_id' in tweet['geo']:
                            tweet['location'].update(places_info[tweet['geo']['place_id']])
                    except KeyError:
                        n_failed_location += 1
                    
                    if tweet['author_id'] in users_info:
                        tweet['username'] = users_info[tweet['author_id']]['username']
                        for k in tweet['location'].keys():
                            tweet['location'][k] = users_info[tweet['author_id']].get(k, '')

                    # target_keys = ['id', 'text', 'created_at', 'location', 'username', 'public_metrics']
                    target_keys = ['id', 'text', 'created_at', 'username']
                    tweet = {k: tweet[k] for k in target_keys}

                    if tweet['id'] in seen_ids:
                        n_duplicate_ids += 1
                        continue
                    
                    tweet_ym = tweet['created_at'][:7]
                    if len(target_yms) > 0 and tweet_ym not in target_yms:
                        continue

                    doc_tweets.append(tweet)
                    username_counter[tweet['username']] += 1
                    seen_ids.add(tweet['id'])

                # finished processed doc tweets
            
            # writing tweets
            if len(doc_tweets) > 0:
                doc_tweets_str = '\n'.join(map(json.dumps, doc_tweets))
                merged_f.write(doc_tweets_str+'\n')
                n_tweets += len(doc_tweets)

    logging.info('Completed')
    logging.info('n_tweets: %d' % n_tweets)
    logging.info('n_usernames: %d' % len(username_counter))
    logging.info('n_failed_location: %d' % n_failed_location)
    logging.info('n_duplicate_ids: %d\n' % n_duplicate_ids)

    # # writing usernames
    # usernames_fn = 'data/usernames.tsv'
    # with open(usernames_fn, 'w') as usernames_f:
    #     for username, count in username_counter.most_common():
    #         usernames_f.write('%s\t%d\n' % (username, count))
