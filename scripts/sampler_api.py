"""
Retrieves tweets for every hour of every day of the given YYYY-MM at the specified MIN_MARK.
For use with Twitter Academic API.

Requires:
Twitter API BEARER_TOKEN set as environment variable.
To set your environment variables in your terminal run the following line:
$ export 'BEARER_TOKEN'='<your_bearer_token>'

Usage:
$ python scripts/sampler_api.py <YYYY> <MM> <MIN_MARK>
$ python scripts/sampler_api.py 2020 01 35

Notes:
MIN_MARK - Specific minute for requests. To allow for multiple requests distanced by user-defined minutes (e.g. every 5 minutes).
"""

import sys
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

search_url = "https://api.twitter.com/2/tweets/search/all"

sleep_duration = 7
retry_duration = 60 * 1 + 1
responses_folder = 'data/responses/'

# stopwords in query selected as top 10 from:
# https://github.com/first20hours/google-10000-english/raw/master/google-10000-english.txt
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

    target_year = int(sys.argv[1])
    target_month = int(sys.argv[2])
    target_min = sys.argv[3]
    target_min = int(target_min)
    assert target_year > 2008
    assert target_month >= 1 and target_month < 13
    assert target_min >= 0 and target_min < 60

    all_periods = []
    for day in range(1, 31+1):

        day_periods = []
        for hour in range(0, 23+1):

            if check_invalid_date(target_year, target_month, day, hour):
                continue

            end_time = datetime(target_year, target_month, day, hour, target_min, 1)
            start_time = end_time - timedelta(hours=1)

            start_time = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_time = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

            day_periods.append((start_time, end_time))
        
        if len(day_periods) > 0:
            all_periods.append(day_periods)


    # check responses already collected
    responses_collected = set()
    for fn in listdir(responses_folder):
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
                    with open(responses_folder + response_fn, 'w') as jl_f:
                        json.dump(wrapped_response, jl_f, indent=4)

                    logging.info('\tSleeping %f secs ...' % sleep_duration)
                    time.sleep(sleep_duration)
                    n_requests_until_fail += 1
                    break

                except Exception as e:
                    logging.info('\tRequest Failed - ', e)
                    logging.info('\t# requests until fail:', n_requests_until_fail)
                    logging.info('\tSleeping %d secs ...' % retry_duration)
                    time.sleep(retry_duration)
                    n_requests_until_fail = 0
                    sleep_duration += 0.01
