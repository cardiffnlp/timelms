"""
Retrieves list of verified Twitter usernames (for preprocessing).

Based on https://raw.githubusercontent.com/twitterdev/Twitter-API-v2-sample-code/main/Follows-Lookup/followers_lookup.py
"""

import os
import json
import time
import requests
from datetime import datetime

# To set your environment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
bearer_token = os.environ.get("BEARER_TOKEN")

def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {bearer_token}"
    r.headers["User-Agent"] = "v2FollowersLookupPython"
    return r


if __name__ == "__main__":

    wait_secs = 60
    wait_secs_on_error = 120

    endpoint = "https://api.twitter.com/2/users/63796828/following"  # @verified

    params = {"user.fields": "created_at,verified,public_metrics", "max_results": 500}

    users = set()
    with open('data/verified_users.jl', 'w') as out_f:
        while True:

            response = requests.request("GET", endpoint, auth=bearer_oauth, params=params)

            if response.status_code != 200:
                print(datetime.now(), "Request returned an error: {} {}".format(response.status_code, response.text))
                print(datetime.now(), "Sleeping %d secs ..." % wait_secs_on_error)
                time.sleep(wait_secs_on_error)
                continue

            json_response = response.json()

            for user_info in json_response['data']:
                out_f.write(json.dumps(user_info)+'\n')
                users.add(user_info['username'])

            try:
                params['pagination_token'] = json_response['meta']['next_token']
            except KeyError:
                break  # end of list

            print(datetime.now(), 'Retrieved %d users.' % len(users))
            time.sleep(wait_secs)
