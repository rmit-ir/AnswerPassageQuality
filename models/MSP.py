import argparse
import itertools
import json
import math
import numpy as np
import string
import sys

from collections import Counter
from krovetzstemmer import Stemmer
from scipy import stats
from smart_open import smart_open
from tqdm import tqdm

from srp.features import qbsum


def MSPLength(query, docs, freqstats, passage_size, increment, lambda_c, min_dl, max_dl):
    """Compute MSP[length] score (Bendersky and Kurland, 2009)

    Args:
        query: query stems, as a list
        docs: list of document stems, as a list of lists
        freqstats: the freqstats object (from srp.features.qbsum)
        passage_size: passage size, in number of stems
        increment: step size when moving the passage window, in number of stems
        lambda_c: mixture parameter
        min_dl: minimum document length
        max_dl: maximum document length
    """
    coll_len = freqstats(None)[0]
    log_min_dl, log_max_dl = math.log(min_dl), math.log(max_dl)

    query_tf = Counter(query)
    query_ctf = {stem: freqstats(stem)[0] for stem in query}

    for doc in docs:
        doc_tf = Counter(doc)
        doc_len = len(doc)

        h = 1.0 - (math.log(doc_len) - log_min_dl) / (log_max_dl - log_min_dl)
        lambda_d, lambda_p = (1 - lambda_c) * h, (1 - lambda_c) * (1 - h)
        msp_score = float('-inf')
        for offset in xrange(0, max(0, doc_len - passage_size) + 1, increment):
            passage = doc[offset:offset+passage_size]
            passage_tf = Counter(passage)
            passage_len = len(passage)

            score = 0
            for stem, qtf in query_tf.items():
                term_score = (lambda_c * query_ctf[stem] / float(coll_len) +
                              lambda_d * doc_tf[stem] / float(doc_len) +
                              lambda_p * passage_tf[stem] / float(passage_len))
                if term_score == 0:
                    score = float('-inf')
                    break
                score += qtf * math.log(term_score)
            if score > msp_score:
                msp_score = score
        yield msp_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='maximum-scoring passage algorithms')
    parser.add_argument('--lambda-c', type=float,
                        help='parameter lambda_c (default: %(default)s)')
    parser.add_argument('--passage-size', type=int)
    parser.add_argument('--increment', type=int)
    parser.add_argument('corpus_json')
    parser.add_argument('freqstats')
    parser.add_argument('min_dl', type=int)
    parser.add_argument('max_dl', type=int)
    parser.set_defaults(passage_size=50, increment=10)
    args = parser.parse_args()

    STEMMER = Stemmer()
    PUNCTUATION_REMOVER = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def tokenize(text):
        terms = text.encode('utf8', errors='replace').translate(PUNCTUATION_REMOVER).lower().split()
        return map(STEMMER, terms)

    print >>sys.stderr, 'Load', args.freqstats
    freqstats = qbsum.make_freqstats(smart_open(args.freqstats))

    coll_len, _ = freqstats(None)

    print >>sys.stderr, 'Load', args.corpus_json
    ranked_lists = json.load(smart_open(args.corpus_json))

    batches = []
    for rl in ranked_lists:
        qid = rl['topic']['qid']
        query = tokenize(rl['topic']['title'])
        docs = []
        metadata = []
        for k, grp in itertools.groupby(rl['sentences'], lambda x: x['docno']):
            docs.append(list(itertools.chain(*[tokenize(x['text']) for x in grp])))
            metadata.append({'qid': qid,
                             'docno': k[:k.rfind('-')],
                             'rel': max(0, rl['topic']['rels'][k]),
                             'score': rl['topic']['scores'][k]})
        batches.append({'qid': qid, 'query': query, 'docs': docs, 'metadata': metadata})

    if args.lambda_c:
        for batch in tqdm(batches, 'Compute MSP for lambda_c = {}'.format(args.lambda_c), unit=''):
            msp_scores = list(MSPLength(batch['query'],
                                        batch['docs'],
                                        freqstats=freqstats,
                                        passage_size=args.passage_size,
                                        increment=args.increment,
                                        lambda_c=args.lambda_c,
                                        min_dl=args.min_dl,
                                        max_dl=args.max_dl))
            for m, msp_score in zip(batch['metadata'], msp_scores):
                print '{rel} qid:{qid} 1:{score} 2:{msp_score} # {docno}'.format(msp_score=msp_score, **m)
    else:
        # for lambda_c in (0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10):
        for lambda_c in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            tau_values = []
            for batch in tqdm(batches, 'Compute MSP for lambda_c = {}'.format(lambda_c), unit=''):
                msp_scores = list(MSPLength(batch['query'],
                                            batch['docs'],
                                            freqstats=freqstats,
                                            passage_size=args.passage_size,
                                            increment=args.increment,
                                            lambda_c=lambda_c,
                                            min_dl=args.min_dl,
                                            max_dl=args.max_dl))
                tau = stats.kendalltau([m['rel'] for m in batch['metadata']], msp_scores)[0]
                tau_values.append(tau)
            print '{}\t{}'.format(lambda_c, np.nanmean(tau_values))
