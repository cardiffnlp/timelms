"""
Retrieves generic tweets for every hour of every day of the given YYYY-MM at the specified MIN_MARK.
For use with Twitter Academic API.

Requires:
Twitter API BEARER_TOKEN set as environment variable.
To set your environment variables in your terminal run the following line:
$ export 'BEARER_TOKEN'='<your_bearer_token>'

Usage:
sampler_api.py [-h] -year YEAR -month MONTH -min_mark MIN_MARK [-dir DIR] [-sleep_duration SLEEP_DURATION]
                    [-retry_duration RETRY_DURATION]

Retrieves generic tweets for every hour of every day of the given YYYY-MM at the specified MIN_MARK.

optional arguments:
  -h, --help            show this help message and exit
  -year YEAR            Target year.
  -month MONTH          Target month.
  -min_mark MIN_MARK    Target minute. To allow for multiple requests distanced by user-defined minutes (e.g., every 5 minutes).
  -dir DIR              Directory for storing responses.
  -sleep_duration SLEEP_DURATION
                        How many seconds to wait between requests.
  -retry_duration RETRY_DURATION
                        How many seconds to wait after failed request.

Example:
$ python scripts/sampler_api.py -year 2020 -month 1 -min_mark 35
"""

import argparse
import os
import json
import time
import requests
import logging
from datetime import datetime
from datetime import timedelta
from os import listdir


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')



bearer_token = os.environ.get("BEARER_TOKEN")


# stopwords in query selected as top 10 from:
# https://github.com/first20hours/google-10000-english/raw/master/google-10000-english.txt
search_url = "https://api.twitter.com/2/tweets/search/all"
query_params = {}
query_params['query'] = '("the" OR "of" OR "and" OR "to" OR "a" OR "in" OR "for" OR "is" OR "on" OR "that") lang:en -is:retweet -is:quote -has:media -has:links -is:nullcast'
query_params['expansions'] = 'author_id,geo.place_id'
query_params['tweet.fields'] = 'id,text,created_at,geo,public_metrics,possibly_sensitive'
query_params['place.fields'] = 'id,full_name,name,country,geo'
query_params['user.fields'] = 'location'
query_params['max_results'] = 500

headers = {}
headers['Authorization'] = "Bearer {}".format(bearer_token)


def query_twitter(search_url, headers, params):
    response = requests.request("GET", search_url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def check_invalid_date(year, month, day, hour):

    try:
        dt = datetime(year, month, day, hour)
    except ValueError:
        return True

    dt_now = datetime.now()
    dt_diff = dt_now.timestamp() - dt.timestamp()
    if dt_diff < (90 * 60):  # min 1h30m diff
        return True
    
    return False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Retrieves generic tweets for every hour of every day of the given YYYY-MM at the specified MIN_MARK.')
    parser.add_argument('-year', help='Target year.', required=True, type=int)
    parser.add_argument('-month', help='Target month.', required=True, type=int)
    parser.add_argument('-min_mark', help='Target minute. To allow for multiple requests distanced by user-defined minutes (e.g., every 5 minutes).', required=True, type=int)
    parser.add_argument('-dir', help='Directory for storing responses.', default='data/responses/', required=False, type=str)
    parser.add_argument('-sleep_duration', help='How many seconds to wait between requests.', default=10, required=False, type=int)
    parser.add_argument('-retry_duration', help='How many seconds to wait after failed request.', default=61, required=False, type=int)
    args = parser.parse_args()

    assert args.year >= 2006
    assert args.month >= 1 and args.month < 13
    assert args.min_mark >= 0 and args.min_mark < 60

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)


    all_periods = []
    for day in range(1, 31+1):

        day_periods = []
        for hour in range(0, 23+1):

            if check_invalid_date(args.year, args.month, day, hour):
                continue

            end_time = datetime(args.year, args.month, day, hour, args.min_mark, 1)
            start_time = end_time - timedelta(hours=1)

            start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_time = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            day_periods.append((start_time, end_time))
        
        if len(day_periods) > 0:
            all_periods.append(day_periods)


    # check responses already collected
    responses_collected = set()
    for fn in listdir(args.dir):
        if fn.endswith('.response.json'):
            responses_collected.add(fn)


    n_requests_until_fail = 0
    for day_periods in all_periods:
        for start_time, end_time in day_periods:
            
            response_fn = '%s_%s.response.json' % (start_time.replace(':', ''), end_time.replace(':', ''))
            
            if response_fn in responses_collected:
                print('Found %s ...' % response_fn)
                continue

            while True:

                try:
                    logging.info('Requesting %s - %s ...' % (start_time, end_time))
                    query_params['start_time'] = start_time
                    query_params['end_time'] = end_time
                    twitter_response = query_twitter(search_url, headers, query_params)
                    wrapped_response = {'start_time': start_time, 'end_time': end_time, 'response': twitter_response}

                    logging.info('\tResults Count: %d' % twitter_response['meta']['result_count'])

                    logging.info('\tWriting %s ...' % response_fn)
                    with open(args.dir + response_fn, 'w') as jl_f:
                        json.dump(wrapped_response, jl_f, indent=4)

                    logging.info('\tSleeping %f secs ...' % args.sleep_duration)
                    time.sleep(args.sleep_duration)
                    n_requests_until_fail += 1
                    break

                except Exception as e:
                    logging.info('\tRequest Failed - ', e)
                    logging.info('\t# requests until fail:', n_requests_until_fail)
                    logging.info('\tSleeping %d secs ...' % args.retry_duration)
                    time.sleep(args.retry_duration)
                    n_requests_until_fail = 0
                    args.sleep_duration += 0.01
