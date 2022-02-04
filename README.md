# TimeLMs: Diachronic Language Models from Twitter

TimeLMs allows for easy access to models continuously trained on social media over regular intervals for researching language model degradation, as well as cultural shifts affecting language usage on social media.

Paper: [https://arxiv.org/pdf/TBD.pdf](https://arxiv.org/pdf/TBD.pdf)

Below we provide instructions for getting started with TimeLMs and a few usage examples.

For a more detailed guide, please see our [notebook demo](demo.ipynb).


# Getting Started

## Environment and Dependencies

You may create a new environment using conda and install dependencies following the commands below.
We assume you already have PyTorch with CUDA support installed (tested with torch==1.8.2+cu111 and CUDA 11.2).

```bash
$ conda create -n timelms python=3.7
$ conda activate timelms
$ pip install -r requirements.txt
```


## Loading TimeLMs

You can load our interface simply with these two lines, importing the TimeLMs class from the [timelms.py](timelms.py) file in this repository.

```python
from timelms import TimeLMs
tlms = TimeLMs(device='cuda:0')
```


## Operating Modes

TimeLMs currently supports the following temporal modes for determining which models are employed for different tweets.
1. 'latest': using our most recently trained Twitter model.
2. 'YYYY-MM' (custom): using the model closest to a custom date provided by the user (e.g., '2020-11').
3. 'corresponding': using the model that was trained only until to each tweet's date (i.e., its specific quarter).
4. 'quarterly': using all available models trained over time in quarterly intervals.

The `corresponding` mode requires tweets with a `created_at` field with dates under any format that begins with YYYY-MM.


## Computing Perplexity

```python
tweets = [{'text': 'She is pure heart #SanaTheBBWinner', 'created_at': '2020-02-09T05:55:00.000Z'},
          {'text': 'Looking forward to watching Squid Game tonight !', 'created_at': '2021-10-11T12:34:56.000Z'}]

pseudo_ppls = tlms.get_pseudo_ppl(tweets, mode='corresponding')
```

To get pseudo-perplexity (PPPL) scores for a set of tweets, you just need to pass a list of tweets to `tlms.get_pseudo_ppl()`, specifying your desired mode. Depending on the chosen mode, you'll get a score from each applicable model (2 models for this example). Besides PPPL scores by model, this method also returns the input tweets with their specific pseudo-log likelihood (PLL) values.


## Masked Predictions

```python
tweets = [{"text": "So glad I'm <mask> vaccinated ."},
          {"text": "Looking forward to watching <mask> Game tonight !"}]

preds = tlms.get_masked_predictions(tweets, mode='quarterly', top_k=3)
```

To get masked predictions using our models, you just need to pass a list of tweets to `tlms.get_masked_predictions()`, specifying your desired mode and number of predictions.
In the example above, we're choosing the `quarterly` mode, which does not require date fields.


## Evaluating Models

```python
tlms.eval_model('roberta-base', 'data/tweets/tweets-2020-2021-subset-rnd.jl')
```

We also provide a method for evaluating other models supported by the Transformers package using PPPL.
For evaluating over the periods of 2020 to 2021, we recommend retrieving the tweets used for our evaluation (we provide tweet ids [here](data/test_ids.csv)), or using the 50K subset provided in this repository as an alternative.
For the time being, we only support models based on RoBERTa (most Twitter LMs).

# Creating Twitter Corpora

Below you find instructions for using our scripts to retrieve and preprocess Twitter data.
The same scripts were used for obtaining our training and testing corpora for TimeLMs.

## Sampling Tweets from the API

```bash
$ python scripts/sampler_api.py 2020 01 35  # <YYYY> <MM> <MIN_MARK>
```

A generic sample of tweets from the Twitter API can be retrieved using the [sampler_api.py](scripts/sampler_api.py) script.
By 'generic' we mean tweets that are not targetting any specific content (we use stopwords as query terms, more details in the paper).

The MIN_MARK variable is the specific minute passed to the API request. You should set this value according to your preference for the time difference between requests. In our paper, we used several calls to this script in increments of 5 minutes.

This script retrieves tweets for every hour of every day of the given YYYY-MM at the specified MIN_MARK.
Every response is stored as its own file in `data/responses`. Requests for files already in that folder will be skipped.

Requires the API BEARER_TOKEN available as an environment variable. You can set that up with:

```bash
$ export 'BEARER_TOKEN'='<your_bearer_token>'
```

The script is set up to wait 7 seconds between requests of 500 results. In case of error, the scripts waits another 60 seconds before retrying (and increments time between requests by 0.01 seconds).

## Compiling Tweets by Date

```bash
$ python scripts/combine.py tweets-2020-Q3.jl 2020-01 2020-02 2020-03  # <output_file> <months:YYYY-MM>
```

After populating data/responses with tweets retrieved from the API, you can use the [combine.py](scripts/combine.py) script to combine those responses into a single .jl file restricted to tweets for specified year-months.

This script also merges metrics and location info so that all data pertaining to a particular tweet is contained in a single-line JSON entry of the output .jl file.

You may specify any number of YYYY-MMs. If none are provided, the script will use all available tweets.

## Cleaning, Filtering and Anonymizing

```bash
$ python scripts/preprocess.py tweets-2020-Q3.jl tweets-2020-Q3.cleaned.jl  # <input .jl> <output .jl>
```

Finally, the merged .jl file can be preprocessed using the [preprocess.py](scripts/preprocess.py) script. This step requires the following additional packages:

```bash
$ pip install datasketch==1.5.3
$ pip install xxhash==2.0.2
```

This script removes duplicates, near duplicates and tweets from most frequent users (likely bots, details in the paper) besides replacing user mentions with '@user' for anonymization, except for popular users (i.e., verified users).

The set of verified users was determined using the [get_verified.py](scripts/get_verified.py) script, producing the [verified_users.v310821.txt](data/verified_users.v310821.txt) file shared with this repository.


# License

TimeLMs is released without any restrictions, but our scoring code is based on the [https://github.com/awslabs/mlm-scoring](https://github.com/awslabs/mlm-scoring) repository, which is distributed under [Apache License 2.0](https://github.com/awslabs/mlm-scoring/blob/master/LICENSE). We also refer users to Twitter regulations regarding use of our models and test sets.
