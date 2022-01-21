from collections import defaultdict
from typing import OrderedDict

import transformers
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import pipeline

assert transformers.__version__ == '4.9.2'


class TimeLMs(object):

    def __init__(self, device='cpu'):
        self.device = device
        self.config = {}

        self.config['account'] = None
        self.config['slug'] = None
        self.config['default'] = None
        self.config['latest'] = None
        self.set_config()


    def set_config(self):
        # TO-DO: fetch from an endpoint instead of hardcoding
        self.config['account'] = 'cardiffnlp'
        self.config['slug'] = 'twitter-roberta-base'
        self.config['default'] = 'cardiffnlp/twitter-roberta-base-2021-124m'
        self.config['latest'] = 'cardiffnlp/twitter-roberta-base-dec2021'

        self.config['quarterly'] = []
        for y in ['2020', '2021']:
            for m in ['mar', 'jun', 'sep', 'dec']:
                self.config['quarterly'].append(f"{self.config['account']}/{self.config['slug']}-{m}{y}")


    def date2model(self, date_str):
        # assuming format 2020-01-01T00:00:00.000Z
        try:
            y = int(date_str[:4])
            m = int(date_str[5:7])
            assert int(m) in range(1, 12+1)
            assert int(y) in range(2006, 2099)
        except:
            raise(BaseException('Invalid date format.'))

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
        for tw in tweets:

            if mode == 'default':
                tw_model = self.config['default']
                tweets_by_model[tw_model].append(tw)
            
            elif mode == 'latest':
                tw_model = self.config['latest']
                tweets_by_model[tw_model].append(tw)

            elif mode == 'specific':
                tw_model = self.date2model(tw['created_at'])
                tweets_by_model[tw_model].append(tw)
            
            elif mode == 'quarterly':
                for tw_model in self.config['quarterly']:
                    tweets_by_model[tw_model].append(tw)
            
            else:
                raise(BaseException("Invalid mode (choose 'default', 'latest', 'specific', 'quarterly')."))
        
        return tweets_by_model


    def get_masked_predictions(self, tweets, mode='default', top_k=3, verbose=False):

        def make_masked_pipeline(model_name):

            if verbose:
                print('Loading %s ...' % model_name)

            model = AutoModelForMaskedLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            if self.device != 'cpu':
                model.to(self.device)
            
            model.eval()

            return pipeline('fill-mask', model=model, tokenizer=tokenizer, device=0)


        tweets_by_model = self.group_tweets_by_model(tweets, mode)

        for model_name, model_tweets in tweets_by_model.items():
            
            pipe = make_masked_pipeline(model_name)
            all_model_preds = pipe([tw['text'] for tw in model_tweets], top_k=top_k)

            if len(model_tweets) == 1:  # quick-fix: pipe() appears to change output shape if just 1
                all_model_preds = [all_model_preds]

            for tw, preds in zip(model_tweets, all_model_preds):
                if mode in ['quarterly']:
                    if 'predictions' not in tw:
                        tw['predictions'] = {model_name: preds}
                    else:
                        tw['predictions'][model_name] = preds
                else:
                    tw['predictions'] = {model_name: preds}

        return tweets


    def get_pseudo_ppl(self, tweets, mode='default', verbose=False):

        from pseudo_ppl import score
        
        tweets_by_model = self.group_tweets_by_model(tweets, mode)

        pppl_by_model = OrderedDict()
        for model_name, model_tweets in tweets_by_model.items():
            pseudo_ppl = score(model_name, model_tweets, mode=mode, verbose=verbose)
            pppl_by_model[model_name] = {'pppl': pseudo_ppl, 'n_tweets': len(model_tweets)}

        return pppl_by_model
