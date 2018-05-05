from __future__ import print_function

import argparse
import json
import logging

from smart_open import smart_open
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract passages from corpus json file')
    parser.add_argument('corpus_json')
    parser.add_argument('output')
    args = parser.parse_args()

    output = smart_open(args.output, 'wb')
    logging.info('output is sent to {}'.format(args.output))

    ranked_lists = json.load(smart_open(args.corpus_json))
    for rl in tqdm(ranked_lists, desc='process ranked lists'):
        qid = rl['topic']['qid']
        for doc in rl['docs']:
            docno = doc['docno']

            sentences = doc['sentences']
            sentence_lengths = [len(sent.split()) for sent in sentences]

            passages = []
            passage_size = 50
            i = 0
            sz = 0
            for j in range(1, len(sentences)):
                sz += sentence_lengths[j]
                if sz >= passage_size:
                    psg = ' '.join(sentences[i:j+1])
                    if len(psg) and sz <= 4 * passage_size:
                        passages.append(psg)
                    while sz >= 0.5 * passage_size:
                        sz -= sentence_lengths[i]
                        i += 1

            res = {'id': '{}.{}'.format(qid, docno), 'text': '\n\n'.join(passages)}
            print(json.dumps(res, sort_keys=True), file=output)
