from __future__ import print_function

import argparse
import collections
import itertools
import json
import krovetzstemmer
import logging
import math
import string

import fastText
import numpy as np
import scipy.stats.mstats
import scipy.special
import sklearn

from smart_open import smart_open
from ortools.linear_solver import pywraplp
from tqdm import tqdm


# Tokenizer
#
PUNCTUATION_REMOVER = string.maketrans(string.punctuation, ' ' * len(string.punctuation))


def to_terms(text):
    return text.encode('utf8', errors='replace').translate(PUNCTUATION_REMOVER).split()


class Tokenizer(object):
    def __init__(self, stemmer=None, stoplist=None):
        self.stemmer = stemmer if stemmer else krovetzstemmer.Stemmer()
        self.stoplist = stoplist if stoplist else set()

    def __call__(self, text):
        return [self.stemmer(term) for term in to_terms(text) if term not in self.stoplist]


# Feature extraction
#
class Pipeline():
    """Feature extraction pipeline"""

    def __init__(self):
        self.jobs = []

    def add(self, features, adaptor=None):
        """Add a group of features into pipeline, optionally under adapted input

        Args:
            features: one or a list of callable feature objects
            adaptor: callable object to transform input data
        """
        if not isinstance(features, (tuple, list)):
            features = [features]
        self.jobs.append({'adaptor': adaptor, 'extractors': features})

    def extract(self, item):
        """Execute pipeline to extract features from item

        Args:
            item: input data
        """
        vector = []
        for job in self.jobs:
            input_ = item if job['adaptor'] is None else job['adaptor'](item)
            for extractor in job['extractors']:
                vector.append(extractor(input_))
        return vector


# Quality features
#
def get_bigrams(sentences):
    for sentence in sentences:
        for bigram in itertools.izip(sentence, sentence[1:]):
            yield bigram


def CQAOverlap(doc):
    model, sentences = doc
    gold = set([b for b in get_bigrams(model['answers'])])
    system = set([b for b in get_bigrams(sentences)])
    return float(len(gold & system)) / len(gold)


def NumSentences(doc):
    _, sentences = doc
    return len(sentences)


def QueryOverlap(doc):
    model, terms = doc
    query = set(model['query'])
    return sum(int(t in query) for t in terms)


def AvgWordWeight(doc):
    model, terms = doc
    return float(sum(model['weights'](t) for t in terms)) / len(terms) if terms else 0


def AvgTermLen(doc):
    """Average length of visible term on the page"""
    _, terms = doc
    return float(sum(len(t) for t in terms)) / len(terms) if terms else 0


def Entropy(doc):
    """Entropy of the page content"""
    _, terms = doc
    N = len(terms)
    tf = collections.Counter(terms)
    return math.log(N) - float(sum(n * math.log(n) for n in tf.values())) / N if N > 0 else 0


class FracStops():
    """Stopword/non-stopword ratio"""
    def __init__(self, stoplist):
        self.stoplist = stoplist

    def __call__(self, doc):
        _, terms = doc
        return float(sum(term in self.stoplist for term in terms)) / len(terms) if terms else 0


class StopCover():
    """Fraction of terms in the stopword list that appear on the page"""
    def __init__(self, stoplist):
        self.stoplist = stoplist

    def __call__(self, doc):
        _, terms = doc
        if self.stoplist:
            return float(sum(sw in terms for sw in self.stoplist)) / len(self.stoplist)
        else:
            return 0


def load_freqstats(iterable):
    tf, df = {}, {}
    for line in iterable:
        term, tf_value, df_value = line.split()
        tf[term] = int(tf_value)
        df[term] = int(df_value)
    N, D = tf.pop("TOTAL"), df.pop("TOTAL")
    return N, D, tf, df


def load_idf(iterable):
    idf = {}
    for line in iterable:
        term, value = line.split()
        idf[term] = float(value)
    return idf


def load_dr_data(iterable):
    res = {}
    for line in iterable:
        payload = json.loads(line)
        bag = {}
        for docno in payload['passages'].keys():
            passage = payload['passages'][docno]
            score = payload['psg_scores'][docno]

            # FIXME: quick and dirty
            sentences = [p + ' .' for p in passage.split(' . ')]
            tokenized_sentences = [s.split() for s in sentences]
            bag[docno] = {'passage': tokenized_sentences, 'passage_text': sentences, 'objective': score}
        res[payload['qid']] = bag
    return res


# Answer relevance models
#
def UniformRelevance(n):
    return np.ones(n) / n


def GradedRelevance(n):
    return 1 / np.log(np.arange(1, n + 1) + 1)


# Term relevance estimates
#
class Thingy(sklearn.base.BaseEstimator):
    pass


class QLEstimator(Thingy):
    """Estimating p(t|A) using query likelihood"""
    def __init__(self, N, ctf, mu):
        assert mu >= 0

        self.N = N
        self.ctf = ctf
        self.mu = mu

    def get_estimate(self, doc, tf):
        dl = sum(tf.values())

        def _zero_estimate(t):
            return 0
        def _estimate(t):
            if t not in tf:
                return 0
            p_background = float(self.ctf.get(t, 0)) / self.N if self.N > 0 else 0
            return float(tf[t] + self.mu * p_background) / (dl + self.mu)

        return _estimate if dl + self.mu > 0 else _zero_estimate


class BM25Estimator(Thingy):
    """Estimating p(t|A) using BM25"""
    def __init__(self, D, df, k1, b, avg_dl):
        self.D = D
        self.df = df
        self.k1 = k1
        self.b = b
        self.avg_dl = avg_dl

    def get_estimate(self, doc, tf):
        dl = sum(tf.values())

        def _estimate(t):
            res = 0
            if t in tf and t in self.df:
                freq = tf[t]
                idf = math.log(float(self.D) / self.df[t])
                res = idf * ((freq * (self.k1 + 1)) /
                             (freq + self.k1 * (1 - self.b + self.b * (dl / self.avg_dl))))
            return res
        return _estimate


class WordEmbeddingEstimator(Thingy):
    """Estimating p(t|A) using word embeddings"""
    def __init__(self, vectors, k, x0):
        self.vectors = vectors
        self.top_words = vectors.get_top_words(n=1000)
        self.k = k
        self.x0 = x0

    def get_estimate(self, doc, tf):
        terms = filter(lambda t: t in self.vectors, tf.keys())  # NOTE: exclude OOV words
        if not terms:
            return lambda t: 0

        freq = np.array([tf[t0] for t0 in terms])
        vec = np.vstack([self.vectors[t0] for t0 in terms])

        def _score(t):
            res = 0
            if t in self.vectors:
                # v = self.vectors[t]
                # values = scipy.special.expit(self.k * (np.array([np.dot(v, v0) for v0 in vec]) - self.x0)) ** freq
                # res = scipy.stats.mstats.gmean(values)
                values = -np.log1p(np.exp(- self.k * (np.dot(vec, self.vectors[t]) - self.x0)))
                res = np.exp(np.dot(values, freq) / freq.sum())
            return res

        Z = sum(_score(t) for t in self.top_words)

        return lambda t: _score(t) / Z


def build_weight_function(answers, relevance, estimator):
    estimates = []
    for answer in answers:
        tf = collections.Counter(answer)
        estimates.append(estimator.get_estimate(answer, tf))
    rels = relevance(len(answers))
    pairs = zip(rels, estimates)

    def _weight_function(t):
        return sum(r * est(t) for r, est in pairs)
    return _weight_function


def build_models(queries, relevance, estimator, tokenizer):
    models = {}
    for query in queries:
        original_query = tokenizer(query['text'])
        answers = [tokenizer(q['best_answer'] or '') for q in query['questions']]
        weights = build_weight_function(answers, relevance, estimator)
        models[query['qid']] = {'query': original_query,
                                'answers': answers,
                                'weights': weights}
    return models


def build_pipeline(stoplist):
    def MODEL_TERMS(doc):
        model, summary = doc
        return model, list(itertools.chain(*summary))

    pipeline = Pipeline()
    pipeline.add([CQAOverlap, NumSentences])
    pipeline.add([QueryOverlap, AvgWordWeight, AvgTermLen, Entropy, FracStops(stoplist), StopCover(stoplist)],
                 adaptor=MODEL_TERMS)
    return pipeline


class PSGExtractor(Thingy):

    def __init__(self, K):
        self.K = K

    def extract(self, doc, tokenizer, weights, alpha):
        tokenized_sentences = map(tokenizer, doc['sentences'])
        obj, idx = find_max_scoring_passages(tokenized_sentences, weights, K=self.K)
        passage = [tokenized_sentences[i] for i in idx]
        passage_text = [doc['sentences'][i] for i in idx]
        return {'objective': obj, 'idx': idx, 'passage': passage, 'passage_text': passage_text}


class ILPExtractor(Thingy):

    def __init__(self, K, M):
        self.K = K
        self.M = M
        self.blacklist = set()

    def add_to_blacklist(self, qid, docno):
        self.blacklist.add((qid, docno))

    def extract(self, doc, tokenizer, weights, alpha):
        # FIXME: work around the Google or-tools segfault problem
        if (doc['qid'], doc['docno']) in self.blacklist:
            return {'objective': 0, 'idx': [], 'passage': [], 'passage_text': []}

        tokenized_sentences = map(tokenizer, doc['sentences'])
        obj, idx = summarize(tokenized_sentences, weights, M=self.M, K=self.K, alpha=alpha)
        passage = [tokenized_sentences[i] for i in idx]
        passage_text = [doc['sentences'][i] for i in idx]
        return {'objective': obj, 'idx': idx, 'passage': passage, 'passage_text': passage_text}


class DRProcessor(Thingy):

    def __init__(self, K, dr_data):
        self.K = K
        self.dr_data = dr_data

    def extract(self, doc, tokenizer, weights, alpha):
        qid, docno = doc['qid'], doc['docno']
        assert qid in self.dr_data, 'Missing qid %s in dr_data' % qid

        if docno in self.dr_data[qid]:
            entry = self.dr_data[qid][docno]
            obj, idx, passage, passage_text = entry['objective'], [], entry['passage'], entry['passage_text']
        else:
            obj, idx, passage, passage_text = 0, [], [], []

        return {'objective': obj, 'idx': idx, 'passage': passage, 'passage_text': passage_text}

    def get_estimate(self, doc, tf):
        def _estimate(t):
            return 0
        return _estimate


class WordEmbeddings(Thingy):
    def __init__(self, fname, engine):
        self.cache = {}
        if engine in ['fastText', 'fasttext']:
            self.model = fastText.load_model(fname)
        else:
            raise ValueError('invalid engine name: {}'.format(engine))

    def __contains__(self, word):
        return self.model.get_word_id(word) != -1

    def __getitem__(self, word):
        if word not in self.cache:
            v = self.model.get_word_vector(word)
            self.cache[word] = v / np.linalg.norm(v, 2)  # NOTE: also normalize the vector
        return self.cache[word]

    def get_top_words(self, n):
        return self.model.get_words()[:n]


def yield_passages(sentences, K):
    sentence_lengths = [len(sentence) for sentence in sentences]
    i, sz = 0, 0
    for j in range(1, len(sentences)):
        assert i <= j
        sz += sentence_lengths[j]
        if sz >= K:
            yield range(i, j + 1)
            while sz >= 0.5 * K:
                sz -= sentence_lengths[i]
                i += 1


def find_max_scoring_passages(sentences, weights, K):
    terms = set()
    for sentence in sentences:
        terms.update(sentence)
    w = {t: weights(t) for t in terms}

    best_score, best_psg_idx = 0, []
    for psg_idx in yield_passages(sentences, K=K):
        score, sz = 0, 0
        for i in psg_idx:
            score += sum(w[t] for t in sentences[i])
            sz += len(sentences[i])
        score = float(score) / sz if sz != 0 else 0
        if score > best_score:
            best_score, best_psg_idx = score, psg_idx
    return best_score, best_psg_idx


def summarize(sentences, weights, M, K, alpha):
    """Maximum coverage summarization algorithm (Takamura & Okumura, 2009)"""
    terms = set()
    for sentence in sentences:
        for t in sentence:
            terms.add(t)
    terms = {t: j for j, t in enumerate(terms)}

    w = {t: weights(t) for t in terms}

    a = {t: [] for t in terms}
    for i, sentence in enumerate(sentences):
        for t in sentence:
            a[t].append(i)

    solver = pywraplp.Solver('SolveIntegerProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    x = [solver.IntVar(0, int(len(sentences[i]) >= M), 'x_{}'.format(i)) for i in range(len(sentences))]
    z = [solver.IntVar(0, 1, 'z_{}'.format(i)) for i in range(len(terms))]

    # Budget constraint
    budget = solver.Constraint(0, K)
    for i, sentence in enumerate(sentences):
        budget.SetCoefficient(x[i], len(sentence))

    # Per-term coverage constraint
    for j, term in enumerate(terms):
        term_coverage = solver.Constraint(-solver.infinity(), 0)
        term_coverage.SetCoefficient(z[j], 1)
        for i in a[term]:
            term_coverage.SetCoefficient(x[i], -1)

    obj = solver.Objective()
    for i, sentence in enumerate(sentences):
        obj.SetCoefficient(x[i], alpha * sum(w[t] for t in sentence))
    for j, term in enumerate(terms):
        obj.SetCoefficient(z[j], (1 - alpha) * w[term])
    obj.SetMaximization()

    result_status = solver.Solve()

    # FIXME: workaround
    if result_status != pywraplp.Solver.OPTIMAL:
        return 0, []

    assert result_status == pywraplp.Solver.OPTIMAL
    obj_value = solver.Objective().Value()
    return obj_value, [i for i, elem in enumerate(x) if elem.solution_value() > 0]


def main():
    parser = argparse.ArgumentParser(description='Extract answer passages')
    parser.add_argument('-alpha', type=float,
                        help='alpha')
    parser.add_argument('-n', type=int,
                        help='produce a summary of at most N words')
    parser.add_argument('-m', type=int,
                        help='only select sentences with more than M words')
    parser.add_argument('-dump-passages', metavar='FILE',
                        help='dump passage output to FILE')

    parser.add_argument('-k1', type=float)
    parser.add_argument('-b', type=float)
    parser.add_argument('-avg-dl', type=float)
    parser.add_argument('-k', type=float)
    parser.add_argument('-x0', type=float)
    parser.add_argument('-idf')
    parser.add_argument('-mu', type=float)
    parser.add_argument('-stoplist')
    parser.add_argument('-freqstats')
    parser.add_argument('-fasttext-bin')
    parser.add_argument('-dr-jsonl')

    parser.add_argument('method')
    parser.add_argument('corpus_json')
    parser.add_argument('ya_json')
    parser.add_argument('output')

    parser.set_defaults(alpha=0.1, n=50, m=5, k1=1.2, b=0.75, avg_dl=100, mu=100, k=10, x0=0)
    args = parser.parse_args()
    
    if args.method not in ['PSG+QL', 'PSG+BM25', 'PSG+Emb', 'DR', 'ILP+QL', 'ILP+BM25', 'ILP+Emb']:
        raise ValueError('invalid method: {}'.format(args.method))

    logging.info('Load queries/YA data')
    queries = json.load(smart_open(args.ya_json))

    logging.info('Set up feature extraction pipeline')
    stoplist = set(l.strip() for l in smart_open(args.stoplist))
    tokenizer = Tokenizer(stoplist=stoplist)
    pipeline = build_pipeline(stoplist=stoplist)

    # use graded relevance by default
    relevance = GradedRelevance

    # assign the extractor based on the method
    if args.method in ['PSG+QL', 'PSG+BM25', 'PSG+Emb']:
        extractor = PSGExtractor(K=args.n)
    elif args.method in ['ILP+QL', 'ILP+BM25', 'ILP+Emb']:
        extractor = ILPExtractor(K=args.n, M=args.m)

        # FIXME: workaround 
        extractor.add_to_blacklist('767', 'GX245-22-3433326')
        extractor.add_to_blacklist('785', 'GX025-94-8770307')
        extractor.add_to_blacklist('837', 'GX268-01-5397177')
        extractor.add_to_blacklist('136', 'clueweb09-enwp01-82-19274')

    # assign the estimator based on the method
    if args.method in ['PSG+QL', 'ILP+QL']:
        if not args.freqstats:
            logging.error('Missing option -freqstats')
            return
        logging.info('Load freqstats')
        N, _, ctf, _ = load_freqstats(smart_open(args.freqstats))
        estimator = QLEstimator(N=N, ctf=ctf, mu=args.mu)

    elif args.method in ['PSG+BM25', 'ILP+BM25']:
        if not args.freqstats:
            logging.error('Missing option -freqstats')
            return
        logging.info('Load freqstats')
        _, D, _, df = load_freqstats(smart_open(args.freqstats))
        estimator = BM25Estimator(D=D, df=df, k1=args.k1, b=args.b, avg_dl=args.avg_dl)

    elif args.method in ['PSG+Emb', 'ILP+Emb']:
        if not args.fasttext_bin:
            logging.error('Missing option -fasttext-bin')
            return
        logging.info('Load fasttext bin')
        vectors = WordEmbeddings(args.fasttext_bin, 'fastText')
        estimator = WordEmbeddingEstimator(vectors=vectors, k=args.k, x0=args.x0)

    # NOTE: DR is configured in a particular way as it primarily loads precomputed data
    if args.method in ['DR']:
        if not args.dr_jsonl:
            logging.error('Missing option -dr-jsonl')
            return
        logging.info('Load dr_jsonl')
        dr_data = load_dr_data(smart_open(args.dr_jsonl))
        extractor = estimator = DRProcessor(K=args.n, dr_data=dr_data)

    print(extractor)
    print(estimator)

    models = build_models(queries,
                          relevance=relevance,
                          estimator=estimator,
                          tokenizer=tokenizer)

    logging.info('Load corpus data (may take a while...)')
    ranked_lists = json.load(smart_open(args.corpus_json))

    logging.info('Will send output to {}'.format(args.output))
    output = smart_open(args.output, 'wb')

    passage_output = None
    if args.dump_passages:
        logging.info('Will send passage output to {}'.format(args.dump_passages))
        passage_output = smart_open(args.dump_passages, 'wb')

    for rl in tqdm(ranked_lists, desc='Process ranked lists'):
        qid = rl['topic']['qid']

        for doc in rl['docs']:
            docno = doc['docno']
            rel = max(doc['rel'], 0)
            score = doc['score']
            logging.debug('Process qid {} docno {}'.format(qid, docno))

            result = extractor.extract(doc,
                                       tokenizer=tokenizer,
                                       weights=models[qid]['weights'],
                                       alpha=args.alpha)

            obj = result['objective']
            answer_passage = result['passage']
            answer_passage_text = result['passage_text']

            if passage_output:
                print(qid, docno, rel, score, file=passage_output)
                for sentence in answer_passage_text:
                    print('    ' + sentence.encode('utf8', errors='ignore'), file=passage_output)

            vector = ' '.join(['{}:{}'.format(i, val) for i, val in
                               enumerate(pipeline.extract((models[qid], answer_passage)), 3)])
            print('{rel} qid:{qid} 1:{score} 2:{obj} {vector} # {docno}'.format(**locals()),
                  file=output)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)
    main()
