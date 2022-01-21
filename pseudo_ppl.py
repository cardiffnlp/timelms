import warnings
# warnings.filterwarnings('ignore')  # TO-DO: resolve userwarnings
warnings.filterwarnings('ignore', message='Some buckets are empty')

import math
import transformers
import mxnet

from mlm.scorers import MLMScorerPT
from mlm.models import RobertaMaskedLMOptimized

assert transformers.__version__ == '4.9.2'


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def score(model_name, tweets, batch_size=16, max_len=256, mode='default', verbose=False):
    """ Changes tweets to include scores by model (and n_subtokens). """
    
    if verbose:
        print('Loading %s ...' % model_name)

    model = RobertaMaskedLMOptimized.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, normalization=True, use_fast=False)
    
    ctxs = [mxnet.cpu()] # or, e.g., [mxnet.gpu(0), mxnet.gpu(1)]
    scorer = MLMScorerPT(model, None, tokenizer, ctxs)
    
    for tw in tweets:
        n_subtokens = len(tokenizer.encode(tw['text']))
        if 'subtokens' not in tw.keys():
            tw['subtokens'] = {model_name: n_subtokens}
        else:
            tw['subtokens'][model_name] = n_subtokens

    # TO-DO: raise warning that we're discarding tweets, or use a default null score
    tweets = [tw for tw in tweets if tw['subtokens'][model_name] <= max_len]

    n_tweets_processed, sum_pll, n_subtokens_total = 0, 0, 0
    for batch in chunks(tweets, batch_size):
        batch_texts = [tw['text'] for tw in batch]
        scores = scorer.score_sentences(batch_texts)

        assert len(batch) == len(scores)
        for tw, tw_pll in zip(batch, scores):
            sum_pll += tw_pll
            n_subtokens_total += tw['subtokens'][model_name]

            if mode in ['quarterly']:
                if 'scores' not in tw.keys():
                    tw['scores'] = {model_name: tw_pll}
                else:
                    tw['scores'][model_name] = tw_pll
            else:
                tw['scores'] = {model_name: tw_pll}

            n_tweets_processed += 1
            if verbose and (len(tweets) > 1000) and (n_tweets_processed % 1000 == 0):
                print(f"[{model_name}] Processed {n_tweets_processed}/{len(tweets)} tweets.")

    pseudo_ppl = math.e**(-1*(sum_pll/n_subtokens_total))

    return pseudo_ppl
    


if __name__ == '__main__':

    tweets = [{'text': 'There are cops all over town.'},
              {'text': 'Bitcoin is going up.'},
              {'text': 'So glad I\'m fully vaccinated.'}]
    
    model_name = 'cardiffnlp/twitter-roberta-base-2021-124m'
    pseudo_ppl = score(model_name, tweets)
    print(pseudo_ppl)  # 2.084653124051384
