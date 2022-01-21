# Based on:
# https://github.com/awslabs/mlm-scoring

from abc import ABC
import logging
from typing import List, Optional, Tuple, Iterable
from collections import OrderedDict

import numpy as np

# MXNet-based
import gluonnlp as nlp
import mxnet as mx
from mxnet.gluon import Block
from mxnet.gluon.data import SimpleDataset

# PyTorch-based
import torch
from . import batchify as btf_generic


class Corpus(OrderedDict):
    """A ground truth corpus (dictionary of ref sentences)
    """

    @classmethod
    def from_text(cls, fp: Iterable[str], max_utts=None):
        corpus = cls()
        # For text files, utterance ID is just the zero-indexed line number
        idx = 0
        for line in fp:
            if max_utts is not None and idx >= max_utts:
                break
            corpus[idx] = line.strip()
            idx += 1
        return corpus


class BaseScorer(ABC):
    """A wrapper around a model which can score utterances
    """

    def __init__(self, model: Block, vocab: nlp.Vocab, tokenizer, ctxs: List[mx.Context], eos: Optional[bool] = None, capitalize: Optional[bool] = None) -> None:
        self._model = model
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._ctxs = ctxs
        self._eos = eos
        self._capitalize = capitalize
        self._max_length = 1024

    def _apply_tokenizer_opts(self, sent: str) -> str:
        if self._eos:
            sent += '.'
        if self._capitalize:
            sent = sent.capitalize()
        return sent

    def _corpus_to_data(self, corpus, split_size, ratio, num_workers: int, shuffle: bool=False):

        # Turn corpus into a dataset
        dataset = self.corpus_to_dataset(corpus)

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=0, shuffle=shuffle)

        # logging.info(batch_sampler.stats())
        dataloader = nlp.data.ShardedDataLoader(dataset, pin_memory=True, batch_sampler=batch_sampler, batchify_fn=self._batchify_fn, num_workers=num_workers, thread_pool=True)

        return dataset, batch_sampler, dataloader

    def _true_tok_lens(self, dataset):

        # Compute sum (assumes dataset is in order; skips are allowed)
        prev_sent_idx = None
        true_tok_lens = []
        for tup in dataset:
            curr_sent_idx = tup[0]
            valid_length = tup[2]
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                true_tok_lens.append(valid_length - 2)

        return true_tok_lens

    def _split_batch(self, batch):
        return zip(*[mx.gluon.utils.split_data(batch_compo, len(self._ctxs), batch_axis=0, even_split=False) for batch_compo in batch])

    def score(self, corpus: Corpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 1, num_workers: int = 10, per_token: bool = False) -> List[float]:

        # ctx_cpu = mx.Context('cpu')

        # Get MXNet data objects
        dataset, batch_sampler, dataloader = self._corpus_to_data(corpus, split_size, ratio, num_workers)

        # Get number of tokens
        true_tok_lens = self._true_tok_lens(dataset)

        # Compute scores (total or per-position)
        if per_token:
            scores_per_token = [[None]*(true_tok_len+2) for true_tok_len in true_tok_lens]
        else:
            scores = np.zeros((len(corpus),))

        sent_count = 0
        # batch_log_interval = 20

        batch_score_accumulation = 1
        batch_sent_idxs_per_ctx = [[] for ctx in self._ctxs]
        batch_scores_per_ctx = [[] for ctx in self._ctxs]

        def sum_accumulated_scores():
            for ctx_idx in range(len(self._ctxs)):
                for batch_sent_idxs, batch_scores in zip(batch_sent_idxs_per_ctx[ctx_idx], batch_scores_per_ctx[ctx_idx]):
                    if per_token:
                        # Slow; only use when necessary
                        for batch_sent_idx, batch_score in zip(batch_sent_idxs, batch_scores):
                            scores_per_token[batch_sent_idx.asscalar()] = batch_score
                    else:
                        np.add.at(scores, batch_sent_idxs.asnumpy(), batch_scores.asnumpy())
                batch_sent_idxs_per_ctx[ctx_idx] = []
                batch_scores_per_ctx[ctx_idx] = []

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch = self._split_batch(batch)

            batch_size = self._batch_ops(batch, batch_sent_idxs_per_ctx, batch_scores_per_ctx, temp, per_token=per_token)

            # Ideally we'd accumulate the scores when possible, but something like the below won't work
            # > scores[sent_idxs] += out
            # See In[21] in https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html.
            # Hence, aggregation is done synchronously, every so often
            # (though batch_score_accumulation = 1 seems best, since bucketing is effective in reducing GPU disparity)
            if len(batch_sent_idxs_per_ctx[0]) == batch_score_accumulation:   
                sum_accumulated_scores()

            # Progress
            sent_count += batch_size
            # if (batch_id+1) % batch_log_interval == 0:
            #     logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))

        # In case there are leftovers
        sum_accumulated_scores()

        if per_token:
            return scores_per_token, true_tok_lens
        else:
            return scores.tolist(), true_tok_lens


    def score_sentences(self, sentences: List[str], **kwargs) -> float:
        corpus = Corpus.from_text(sentences)
        return self.score(corpus, **kwargs)[0]


# TODO: Dedup with BaseScorer's score()
class MLMScorerPT(BaseScorer):
    """For models that need every token to be masked
    """

    def __init__(self, *args, **kwargs):
        self._wwm = kwargs.pop('wwm') if 'wwm' in kwargs else False
        self._lang = kwargs.pop('lang') if 'lang' in kwargs else None
        super().__init__(*args, **kwargs)

        ### PyTorch-based
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        # TODO: This does not restrict to specific GPUs however, use CUDA_VISIBLE_DEVICES?
        # TODO: It also unnecessarily locks the GPUs to each other
        self._model.to(self._device)
        self._model = torch.nn.DataParallel(self._model, device_ids=[0])
        self._model.eval()


    def _ids_to_masked(self, token_ids: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:

        # Here:
        # token_ids = [1 ... ... 1012 1], where 1 = </s>

        token_ids_masked_list = []

        assert (not self._wwm)

        mask_indices = []
        if self._wwm:
            raise NotImplementedError
        else:
            mask_indices = [[mask_pos] for mask_pos in range(len(token_ids))]

        # We don't mask the [CLS], [SEP] for now for PLL
        mask_indices = mask_indices[1:-1]

        mask_token_id = self._tokenizer._convert_token_to_id(self._tokenizer.mask_token)
        for mask_set in mask_indices:
            token_ids_masked = token_ids.copy()
            token_ids_masked[mask_set] = mask_token_id

            if self._wwm:
                raise NotImplementedError
            else:
                token_ids_masked_list.append((token_ids_masked, mask_set))

        return token_ids_masked_list


    def corpus_to_dataset(self, corpus: Corpus) -> SimpleDataset:

        sents_expanded = []

        for sent_idx, sent in enumerate(corpus.values()):
            sent = self._apply_tokenizer_opts(sent)
            ids_original = np.array(self._tokenizer.encode(sent, add_special_tokens=True))

            # Enforce max length
            if len(ids_original) > self._max_length:
                logging.error("Line #{} is too long; will output score of 0 and omit in token counts (but not yet in word counts!)".format(sent_idx+1))
            else:
                ids_masked = self._ids_to_masked(ids_original)
                sents_expanded += [(
                        sent_idx,
                        ids,
                        len(ids_original),
                        mask_set,
                        ids_original[mask_set],
                        1)
                    for ids, mask_set in ids_masked]
                # print([self._tokenizer.convert_ids_to_tokens(sent[1]) for sent in sents_expanded[:3] + sents_expanded[-3:]])

        return SimpleDataset(sents_expanded)


    def score(self, corpus: Corpus, temp: float = 1.0, split_size: int = 2000, ratio: float = 0, per_token: bool = False) -> List[float]:

        assert temp == 1.0

        # Turn corpus into a BERT-ready Dataset
        dataset = self.corpus_to_dataset(corpus)

        # Turn Dataset into Dataloader
        batchify_fn = btf_generic.Tuple(btf_generic.Stack(dtype='int32'), btf_generic.Pad(pad_val=self._tokenizer._convert_token_to_id(self._tokenizer.pad_token), dtype='long'),
                              btf_generic.Stack(dtype='long'), btf_generic.Stack(dtype='long'),
                              btf_generic.Stack(dtype='long'), btf_generic.Stack(dtype='long'))

        # TODO: There is a 'by-design' bug in FixedBucketSampler with num_shards > 0, where it silently reuses the last utterances:
        # https://github.com/dmlc/gluon-nlp/blame/b1b61d3f90cf795c7b48b6d109db7b7b96fa21ff/src/gluonnlp/data/sampler.py#L398
        # batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=len(self._ctxs), shuffle=False)
        # Hence, we use num_shards = 0 and do gluon's split_data
        batch_sampler = nlp.data.sampler.FixedBucketSampler([sent_tuple[2] for sent_tuple in dataset], batch_size=split_size, ratio=ratio, num_shards=0, shuffle=False)

        # logging.info(batch_sampler.stats())

        # dataloader = nlp.data.ShardedDataLoader(dataset, pin_memory=True, batch_sampler=batch_sampler, batchify_fn=batchify_fn, num_workers=num_workers, thread_pool=True)
        dataloader = nlp.data.ShardedDataLoader(dataset, batch_sampler=batch_sampler, batchify_fn=batchify_fn)

        # Compute sum (assumes dataset is in order)
        prev_sent_idx = None
        true_tok_lens = []
        for (curr_sent_idx, _, valid_length, _, _, _) in dataset:
            if curr_sent_idx != prev_sent_idx:
                prev_sent_idx = curr_sent_idx
                true_tok_lens.append(valid_length - 2)

        # Compute scores (total or per-position)
        if per_token:
            scores_per_token = [[None]*(true_tok_len+2) for true_tok_len in true_tok_lens]
        else:
            scores = np.zeros((len(corpus),))

        sent_count = 0
        # batch_log_interval = 20

        batch_score_accumulation = 1
        batch_sent_idxs_per_ctx = [[] for ctx in self._ctxs]
        batch_scores_per_ctx = [[] for ctx in self._ctxs]
        batch_masked_positions_per_ctx = [[] for ctx in self._ctxs]

        def sum_accumulated_scores():
            for ctx_idx in range(len(self._ctxs)):
                for batch_sent_idxs, batch_scores, batch_masked_positions in zip(batch_sent_idxs_per_ctx[ctx_idx], batch_scores_per_ctx[ctx_idx], batch_masked_positions_per_ctx[ctx_idx]):
                    if per_token:
                        # Slow; only use when necessary
                        for batch_sent_idx, batch_score, batch_masked_position in zip(batch_sent_idxs, batch_scores, batch_masked_positions):
                            # scores_per_token[batch_sent_idx.asscalar()][int(batch_masked_position.asscalar())] = batch_score.asscalar().item()
                            scores_per_token[batch_sent_idx][batch_masked_position.cpu().numpy().item()] = batch_score.cpu().numpy().item()
                    else:
                        # np.add.at(scores, batch_sent_idxs.asnumpy(), batch_scores.asnumpy())
                        np.add.at(scores, batch_sent_idxs, batch_scores.cpu().numpy())
                batch_sent_idxs_per_ctx[ctx_idx] = []
                batch_scores_per_ctx[ctx_idx] = []
                batch_masked_positions_per_ctx[ctx_idx] = []

        # For now just predicts the first non-cls token
        for batch_id, batch in enumerate(dataloader):

            batch_size = 0

            for ctx_idx, (sent_idxs, token_ids, valid_length, masked_positions, token_masked_ids, normalization) in enumerate((batch,)):

                ctx = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                batch_size += sent_idxs.shape[0]

                # TODO: Super inefficient where we go from MXNet to NumPy to PyTorch

                with torch.no_grad():

                    token_ids = torch.tensor(token_ids)
                    valid_length = torch.tensor(valid_length)
                    masked_positions = torch.tensor(masked_positions).reshape(-1, 1)
                    token_masked_ids = torch.tensor(token_masked_ids).reshape(-1)

                    token_ids = token_ids.to(ctx)
                    valid_length = valid_length.to(ctx)
                    masked_positions = masked_positions.to(ctx)
                    token_masked_ids = token_masked_ids.to(ctx)

                    split_size = token_ids.shape[0]

                    # if isinstance(self._model.module, RobertaMaskedLMOptimized):   
                    # Because BERT does not take a length parameter
                    alen = torch.arange(token_ids.shape[1], dtype=torch.long)
                    alen = alen.to(ctx)
                    mask = alen < valid_length[:, None]
                    out = self._model(input_ids=token_ids, attention_mask=mask, select_positions=masked_positions)
                    out = out[0].squeeze()
                    # else:
                    #     raise ValueError

                    # TODO: Manual numerically-stable softmax
                    # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
                    # Because we only need one scalar
                    out = out.log_softmax(dim=-1)

                    # Get the probability computed for the correct token
                    # Save the scores at the masked indices
                    batch_sent_idxs_per_ctx[ctx_idx].append(sent_idxs)
                    out = out[list(range(split_size)), token_masked_ids]
                    batch_scores_per_ctx[ctx_idx].append(out)
                    batch_masked_positions_per_ctx[ctx_idx].append(masked_positions)

            # Ideally we'd accumulate the scores when possible, but something like the below won't work
            # > scores[sent_idxs] += out
            # See In[21] in https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html.
            # Hence, aggregation is done synchronously, every so often
            # (though batch_score_accumulation = 1 seems best, since bucketing is effective in reducing GPU disparity)
            if len(batch_sent_idxs_per_ctx[0]) == batch_score_accumulation:   
                sum_accumulated_scores()

            # Progress
            sent_count += batch_size
            # if (batch_id+1) % batch_log_interval == 0:
            #     logging.info("{} sents of {}, batch {} of {}".format(sent_count, len(dataset), batch_id+1, len(batch_sampler)))

        # TODO: Test score accumulation
        # In case there are leftovers
        sum_accumulated_scores()

        if per_token:
            return scores_per_token, true_tok_lens
        else:
            return scores.tolist(), true_tok_lens

