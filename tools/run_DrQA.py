#!/usr/bin/env python3
"""Interactive interface to full DrQA pipeline."""

import argparse
import logging
import json
import torch

from functools import partial
from multiprocessing.pool import ThreadPool
from smart_open import smart_open

from drqa import pipeline
from drqa.retriever import utils


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


class RetrievedDocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, topics=None, strict=True):
        """
        Args:
            tfidf_path: path to saved model file
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        # tfidf_path = tfidf_path or DEFAULTS['tfidf_path']
        # logger.info('Loading %s' % tfidf_path)
        # _, metadata = utils.load_sparse_csr(tfidf_path)
        # self.doc_dict = metadata['doc_dict']
        assert topics

        self.topics = {topic['qid']: topic for topic in topics}
        self.title2qid = {topic['title']: topic['qid'] for topic in topics}

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        assert query in self.title2qid, 'query not found: %s' % query

        qid = self.title2qid[query]
        topic = self.topics[qid]

        results = sorted(topic['scores'].items(), key=lambda x: x[1], reverse=True)[:k]
        doc_ids = ['%s.%s' % (qid, id_) for id_, _ in results]
        doc_scores = [score for _, score in results]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        # return [self.closest_docs(query, k=k) for query in queries]

        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reader-model', type=str, default=None,
                        help='Path to trained Document Reader model')
    parser.add_argument('--retriever-model', type=str, default=None,
                        help='Path to Document Retriever model (tfidf)')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help="String option specifying tokenizer type to use (e.g. 'corenlp')")
    parser.add_argument('--candidate-file', type=str, default=None,
                        help="List of candidates to restrict predictions to, one candidate per line")
    parser.add_argument('--no-cuda', action='store_true',
                        help="Use CPU only")
    parser.add_argument('--gpu', type=int, default=-1,
                        help="Specify GPU device id to use")
    parser.add_argument('--skip-to', metavar='QID',
                        help='Start from topic QID and skip over all the previous ones')
    parser.add_argument('--use-desc-topics', metavar='FILE',
                        help='Use desc queries pulled from FILE instead')
    parser.add_argument('corpus_json')
    parser.add_argument('output_json')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        logger.info('CUDA enabled (GPU %d)' % args.gpu)
    else:
        logger.info('Running on CPU only.')

    if args.candidate_file:
        logger.info('Loading candidates from %s' % args.candidate_file)
        candidates = set()
        with open(args.candidate_file) as f:
            for line in f:
                line = utils.normalize(line.strip()).lower()
                candidates.add(line)
        logger.info('Loaded %d candidates.' % len(candidates))
    else:
        candidates = None

    logger.info('Loading query topics from %s (may take a while)' % args.corpus_json)
    topics = [rl['topic'] for rl in json.load(smart_open(args.corpus_json))]
    logger.info('Loaded %d topics.' % len(topics))

    if args.skip_to:
        found = None
        for i in range(len(topics)):
            if topics[i]['qid'] == args.skip_to:
                found = i
                break
        if found is None:
            topics = []
        else:
            topics = topics[found:]

    if args.use_desc_topics:
        logger.info('Loading desc topics (to override title queries)')
        desc_topics = {t['qid']: t for t in json.load(smart_open(args.use_desc_topics))}
        for i in range(len(topics)):
            qid = topics[i]['qid']

            logger.info('%s: %s => %s' % (qid, topics[i]['title'], desc_topics[qid]['desc']))
            topics[i]['title'] = desc_topics[qid]['desc']

    logger.info('Initializing pipeline...')
    DrQA = pipeline.DrQA(
        cuda=args.cuda,
        fixed_candidates=candidates,
        reader_model=args.reader_model,
        ranker_config={'class': RetrievedDocRanker, 'options': {'topics': topics}},
        db_config={'options': {'db_path': args.doc_db}},
        tokenizer=args.tokenizer,
        num_workers=16,
        max_loaders=2,
    )

    # ------------------------------------------------------------------------------
    # Drop in to interactive mode
    # ------------------------------------------------------------------------------

    title_queries = [topic['title'] for topic in topics]
    output = smart_open(args.output_json, 'a')
    ranked_lists = []

    for topic in topics:
        predictions = DrQA.process(topic['title'], None, top_n=100, n_docs=100, return_context=True)
        passages = {}
        psg_scores = {}
        for p in predictions:
            docno = p['doc_id'][p['doc_id'].find('.') + 1:]
            passages[docno] = p['context']['text']
            psg_scores[docno] = p['span_score']

        res = {'qid': topic['qid'],
               'title': topic['title'],
               'scores': topic['scores'],
               'psg_scores': psg_scores,
               'passages': passages}
        ranked_lists.append(res)
        logger.info('qid %s, %d passage scores returned' % (res['qid'], len(res['psg_scores'])))
        payload = json.dumps(res)
        print(payload, file=output)


if __name__ == '__main__':
    main()
