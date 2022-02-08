import json
from collections import defaultdict
from typing import OrderedDict

import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline

assert transformers.__version__ == '4.9.2'


class TimeLMs(object):

    def __init__(self, device='cpu', keep_verified_users=True):
        self.version = '0.9.2'
        self.device = device

        self.config = {}
        self.config['account'] = 'cardiffnlp'
        self.config['slug'] = None
        self.config['default'] = None
        self.config['latest'] = None
        self.set_config()

        self.verified_users = {}
        if keep_verified_users:
            self.load_verified_users()


    def set_config(self):
        # TO-DO: fetch from an endpoint instead of hardcoding
        self.config['slug'] = 'twitter-roberta-base'
        self.config['default'] = 'cardiffnlp/twitter-roberta-base-2021-124m'
        self.config['latest'] = 'cardiffnlp/twitter-roberta-base-dec2021'

        self.config['quarterly'] = []
        for y in ['2020', '2021']:
            for m in ['mar', 'jun', 'sep', 'dec']:
                self.config['quarterly'].append(f"{self.config['account']}/{self.config['slug']}-{m}{y}")


    def load_verified_users(self, verified_fn='data/verified_users.v310821.txt'):
        self.verified_users = set(open(verified_fn).read().split('\n'))


    def date2model(self, date_str):
        # assuming format 2020-01-01T00:00:00.000Z
        try:
            y = int(date_str[:4])
            m = int(date_str[5:7])
            assert int(m) in range(1, 12+1)
            assert int(y) in range(2006, 2099)
        except:
            # raise(BaseException('Invalid date format.'))
            return None

        if m in [1, 2, 3]:
            return f"{self.config['account']}/{self.config['slug']}-mar{str(y)}"
        elif m in [4, 5, 6]:
            return f"{self.config['account']}/{self.config['slug']}-jun{str(y)}"
        elif m in [7, 8, 9]:
            return f"{self.config['account']}/{self.config['slug']}-sep{str(y)}"
        elif m in [10, 11, 12]:
            return f"{self.config['account']}/{self.config['slug']}-dec{str(y)}"


    def model2date(self, model_name):
        # assuming format cardiffnlp/twitter-roberta-base-mar2020
        month_mapper = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                        'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                        'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

        try:
            y = int(model_name[-4:])
            m = month_mapper[model_name[-7:-4]]
            assert m in range(1, 12+1)
            assert y in range(2006, 2099)
        except:
            raise(BaseException('Invalid model_name format.'))

        return (y, m)


    def group_tweets_by_model(self, tweets, mode='default'):

        tweets_by_model = defaultdict(list)

        if mode == 'default':
            tweets_by_model[self.config['default']] = tweets

        elif mode == 'latest':
            tweets_by_model[self.config['latest']] = tweets

        elif self.date2model(mode) != None:  # custom mode, expects YYYY-MM
            tweets_by_model[self.date2model(mode)] = tweets
        
        elif mode == 'corresponding' or mode == 'specific':  # old version used 'specific'
            for tw in tweets:
                tweets_by_model[self.date2model(tw['created_at'])].append(tw)

        elif mode == 'quarterly':
            for tw_model in self.config['quarterly']:
                tweets_by_model[tw_model] = tweets

        else:
            raise(BaseException("Invalid mode (choose 'default', 'latest', 'corresponding', 'quarterly', 'YYYY-MM')."))
    
        return tweets_by_model


    def preprocess_text(self, text):
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        text_cleaned = []
        for t in text.split():
            t = '@user' if t.startswith('@') and len(t) > 1 and t.replace('@','') not in self.verified_users else t
            t = 'http' if t.startswith('http') else t
            text_cleaned.append(t)
        
        return ' '.join(text_cleaned)


    def get_masked_predictions(self, tweets, mode='default', top_k=3, targets=None, verbose=False):

        def make_masked_pipeline(model_name):

            if verbose:
                print('Loading %s ...' % model_name)

            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if self.device != 'cpu':
                model.to(self.device)
            
            model.eval()

            return pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)


        if top_k == -1:
            top_k = 50265  # vocab_size of our roberta-base models

        tweets_by_model = self.group_tweets_by_model(tweets, mode)

        for model_name, model_tweets in tweets_by_model.items():
            
            pipe = make_masked_pipeline(model_name)
            model_texts = [tw['text'] for tw in model_tweets]
            model_texts = [self.preprocess_text(t) for t in model_texts]
            all_model_preds = pipe(model_texts, top_k=top_k, targets=targets)

            if len(model_tweets) == 1:  # quick-fix: pipe() appears to change output shape if just 1
                all_model_preds = [all_model_preds]

            for tw, preds in zip(model_tweets, all_model_preds):
                
                for p in preds:  # for lighter output
                    del p['sequence']

                if mode in ['quarterly']:
                    if 'predictions' not in tw:
                        tw['predictions'] = {model_name: preds}
                    else:
                        tw['predictions'][model_name] = preds
                else:
                    tw['predictions'] = {model_name: preds}

        return tweets


    def get_pseudo_ppl(self, tweets, mode='default', verbose=False):

        from pseudo_ppl import score  # imported on call to allow TimeLMs running without mxnet
        
        tweets_by_model = self.group_tweets_by_model(tweets, mode)

        pppl_by_model = OrderedDict()
        for model_name, model_tweets in tweets_by_model.items():
            pseudo_ppl = score(model_name, model_tweets, mode=mode, verbose=verbose)
            pppl_by_model[model_name] = {'pppl': pseudo_ppl, 'n_tweets': len(model_tweets)}

        return pppl_by_model


    def eval_model(self, model_name, tweets_path, verbose=False):

        from pseudo_ppl import score  # imported on call to allow TimeLMs running without mxnet
        
        # load tweets from given tweets_path
        tweets = []
        with open(tweets_path) as jl_f:
            for jl_str in jl_f:
                tw = json.loads(jl_str)
                tw['text'] = self.preprocess_text(tw['text'])
                tweets.append(tw)

        # model_name can be anything accepted by HF's .from_pretrained()
        pseudo_ppl = score(model_name, tweets, verbose=verbose)

        return {'pppl': pseudo_ppl, 'n_tweets': len(tweets)}

