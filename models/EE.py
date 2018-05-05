import argparse
import itertools
import json
import math
import numpy as np
import pandas
import string
import sys

from collections import Counter, defaultdict
from krovetzstemmer import Stemmer
from scipy import stats
from smart_open import smart_open
from tqdm import tqdm


def make_freqstats(iterable):
    freqstats = {}
    if iterable:
        with tqdm(desc='Load freqstats data (2 steps)') as pbar:
            df = pandas.read_csv(iterable, delim_whitespace=True, names=('term', 'cf', 'df'))
            pbar.update()

            freqstats[None] = (df.cf[0], df.df[0])
            freqstats.update(zip(df.term[1:], zip(df.cf[1:], df.df[1:])))
            del df
            pbar.update()

    def _freqstats(token):
        return freqstats.get(token, (0, 0))
    return _freqstats


def RM3(query, docs, external_docs, freqstats, weights, fb_original_weight, fb_terms,
        dump_term_weights=False):
    """Compute RM3 score with external expansion (Diaz and Metzler, 2006)

    Args:
        query: query stems, as a list
        docs: list of document stems, as a list of lists
        external_docs: list of external document stems, as a list of lists
        freqstats: the freqstats object
        weights: mixture parameter
        fb_original_weight: weight (lambda) of the original query model
        fb_terms: number of terms to mix into the original query model
        dump_term_weights: boolean, also return term distributions if set True
    """
    coll_len = freqstats(None)[0]
    lambda_c, lambda_d = weights

    query_tf = Counter(query)
    query_len = len(query)
    query_ctf = {stem: freqstats(stem)[0] for stem in query}

    prob_w = defaultdict(float)
    for doc in external_docs:
        doc_tf = Counter(doc)
        doc_len = len(doc)

        score = 0
        for stem, qtf in query_tf.items():
            stem_score = (lambda_c * query_ctf[stem] / float(coll_len) +
                          lambda_d * doc_tf[stem] / float(doc_len))
            if stem_score == 0:
                score = float('-inf')
                break
            score += qtf * math.log(stem_score)
        if score == float('-inf'):
            continue

        for stem, dtf in doc_tf.items():
            prob_w[stem] += float(dtf) / doc_len * math.exp(score)

    total_mass = sum(prob_w.values())
    for k, v in prob_w.items():
        prob_w[k] = float(v) / total_mass

    prob_q = {stem: float(qtf) / query_len for stem, qtf in query_tf.items()}
    for stem, p in sorted(prob_w.items(), key=lambda x: x[1], reverse=True)[:fb_terms]:
        prob_q[stem] = fb_original_weight * prob_q.get(stem, 0) + (1 - fb_original_weight) * p

    query_ctf = {stem: freqstats(stem)[0] for stem in prob_q}

    scores = []
    for doc in docs:
        doc_tf = Counter(doc)
        doc_len = len(doc)

        score = 0
        for stem, w in prob_q.items():
            stem_score = lambda_c * query_ctf.get(stem, 1) / float(coll_len)
            stem_score += lambda_d * doc_tf.get(stem, 0) / float(doc_len) if doc_len > 0 else 0
            if stem_score == 0:
                score = float('-inf')
                break
            score += w * math.log(stem_score)
        if score == float('-inf'):
            score = -1000000.0
        scores.append(score)

    if dump_term_weights:
        return scores, prob_q
    else:
        return scores

        # msp_score = float('-inf')
        # for offset in xrange(0, max(0, doc_len - passage_size) + 1, increment):
        #     passage = doc[offset:offset+passage_size]
        #     passage_tf = Counter(passage)
        #     passage_len = len(passage)

        #     score = 0
        #     for stem, qtf in query_tf.items():
        #         term_score = (lambda_c * query_ctf[stem] / float(coll_len) +
        #                       lambda_d * doc_tf[stem] / float(doc_len) +
        #                       lambda_p * passage_tf[stem] / float(passage_len))
        #         if term_score == 0:
        #             score = float('-inf')
        #             break
        #         score += qtf * math.log(term_score)
        #     if score > msp_score:
        #         msp_score = score
        # yield msp_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RM3')
    parser.add_argument('--num-restarts', type=int)
    parser.add_argument('--weights')
    parser.add_argument('--fb-original-weight', type=float)
    parser.add_argument('--fb-terms', type=int)
    parser.add_argument('--dump-term-weights', action='store_true')
    parser.add_argument('corpus_json')
    parser.add_argument('corpus_cqa_json')
    parser.add_argument('freqstats')
    parser.set_defaults(passage_size=50, increment=10, num_restarts=60)
    args = parser.parse_args()

    STEMMER = Stemmer()
    PUNCTUATION_REMOVER = string.maketrans(string.punctuation, ' ' * len(string.punctuation))

    def tokenize(text):
        terms = text.encode('utf8', errors='replace').translate(PUNCTUATION_REMOVER).lower().split()
        return map(STEMMER, terms)

    print >>sys.stderr, 'Load', args.freqstats
    freqstats = make_freqstats(smart_open(args.freqstats))

    print >>sys.stderr, 'Load', args.corpus_cqa_json
    external_docs = {}
    for topic in json.load(smart_open(args.corpus_cqa_json)):
        qid = topic['qid']
        docs = []
        for q in topic['questions']:
            doc = []
            doc.append(q.get('title', ''))
            doc.append(q.get('body', ''))
            if q['answers']:
                doc.append(q['answers'][0])
            docs.append(tokenize('\n\n'.join(doc)))
        external_docs[qid] = docs

    print >>sys.stderr, 'Load', args.corpus_json
    ranked_lists = json.load(smart_open(args.corpus_json))

    batches = []
    for rl in ranked_lists:
        qid = rl['topic']['qid']
        query = tokenize(rl['topic']['title'])
        docs = []
        metadata = []
        # for k, grp in itertools.groupby(rl['sentences'], lambda x: x['docno']):
        #     docs.append(list(itertools.chain(*[tokenize(x['text']) for x in grp])))
        #     metadata.append({'qid': qid,
        #                      'docno': k[:k.rfind('-')],
        #                      'rel': max(0, rl['topic']['rels'][k]),
        #                      'score': rl['topic']['scores'][k]})
        for doc in rl['docs']:
            docno = doc['docno']
            text = list(itertools.chain(*map(tokenize, doc['sentences'])))
            docs.append(text)
            metadata.append({'qid': qid,
                             'docno': docno,
                             'rel': max(0, rl['topic']['rels'][docno]),
                             'score': rl['topic']['scores'][docno]})
        batches.append({'qid': qid, 'query': query, 'docs': docs,
                        'external_docs': external_docs[qid], 'metadata': metadata})

    if args.weights:
        weights = map(float, args.weights.split())
        fb_original_weight = args.fb_original_weight
        fb_terms = args.fb_terms
        for batch in tqdm(batches,
                          'Compute RM3 with weights {}, orig {}'.format(str(weights), fb_original_weight),
                          unit=''):
            rm3_scores, prob_q = RM3(batch['query'],
                                     batch['docs'],
                                     batch['external_docs'],
                                     freqstats=freqstats,
                                     weights=weights,
                                     fb_original_weight=fb_original_weight,
                                     fb_terms=fb_terms,
                                     dump_term_weights=True)  # note that this was turned on
            if args.dump_term_weights:
                term_weights = ' '.join(['{}:{}'.format(k, v) for k, v in
                                         sorted(prob_q.items(), key=lambda x: x[1], reverse=True)])
                print '{qid}\t{term_weights}'.format(qid=batch['qid'], term_weights=term_weights)
            else:
                for m, rm3_score in zip(batch['metadata'], rm3_scores):
                    print '{rel} qid:{qid} 1:{score} 2:{rm3_score} # {docno}'.format(rm3_score=rm3_score, **m)

    else:
        for _ in range(args.num_restarts):
            l = np.round(np.random.rand(), 1)
            weights = [l, 1 - l]
            fb_original_weight = np.round(np.random.rand(), 1)
            fb_terms = 10 * np.random.randint(10)

            tau_values = []
            for batch in batches:
                flip = np.random.randint(2)
                if flip > 0:
                    rm3_scores = list(RM3(batch['query'],
                                          batch['docs'],
                                          batch['external_docs'],
                                          freqstats=freqstats,
                                          weights=weights,
                                          fb_original_weight=fb_original_weight,
                                          fb_terms=fb_terms))
                    tau = stats.kendalltau([m['rel'] for m in batch['metadata']], rm3_scores)[0]
                    tau_values.append(tau)
            print '{}\t{}\t{}\t{}'.format('\t'.join(map(str, weights)), fb_original_weight, fb_terms, np.nanmean(tau_values))
