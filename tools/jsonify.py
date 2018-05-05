from __future__ import print_function

import argparse
import itertools
import json
import os
import re
import sys

from smart_open import smart_open
from tqdm import tqdm

from text_utils import sanitize, get_paragraphs, SentenceDelimiter


def convert_to_original_docno(docno):
    return docno[:docno.rfind('-')]


def get_docs(iterable, extract_sentences=False):
    """ Parse the TREC format input and generate documents """
    docno_pattern = re.compile(r'<DOCNO>(\S+)</DOCNO>')
    qid_pattern = re.compile(r'<QID>(\S+)</QID>')
    score_pattern = re.compile(r'<SCORE>(\S+)</SCORE>')
    relevance_pattern = re.compile(r'<RELEVANCE>(\S+)</RELEVANCE>')

    delimiter = SentenceDelimiter(corenlp_path=os.environ.get('CORENLP_HOME'))

    metadata = None
    doc, dochdr = None, None  # containers for the doc-level content
    for line in iterable:
        if line.startswith('<'):
            if line.startswith('<DOC>'):
                metadata = {}
                doc = []
            elif line.startswith('</DOC>'):
                text = ''.join(doc).decode('latin1')
                sentences = []
                if extract_sentences:
                    for paragraph in get_paragraphs(sanitize(text)):
                        sentences.extend(list(delimiter.get_sentences(paragraph)))
                yield {'docno': convert_to_original_docno(metadata['docno']),
                       'qid': metadata['qid'],
                       'score': metadata['score'],
                       'rel': metadata['relevance'],
                       'url': ''.join(metadata['dochdr']).decode('latin1').strip(),
                       'text': ''.join(doc).decode('latin1'),
                       'sentences': sentences}
                metadata = None
                doc, dochdr = None, None
            elif line.startswith('<DOCHDR>'):
                dochdr = []
            elif line.startswith('</DOCHDR>'):
                metadata['dochdr'] = dochdr
                dochdr = None
            elif line.startswith('<DOCNO>'):
                m = docno_pattern.match(line)
                metadata['docno'] = m.group(1)
            elif line.startswith('<QID>'):
                m = qid_pattern.match(line)
                metadata['qid'] = m.group(1)
            elif line.startswith('<SCORE>'):
                m = score_pattern.match(line)
                metadata['score'] = float(m.group(1))
            elif line.startswith('<RELEVANCE>'):
                m = relevance_pattern.match(line)
                metadata['relevance'] = int(m.group(1))
            else:
                # treat unrecognized tag as text
                if doc is not None and dochdr is None:
                    doc.append(line)
        else:
            # otherwise the line is either a URL string or text content
            if doc is not None:
                if dochdr is None:
                    doc.append(line)
                elif line != ']]>\n':
                    dochdr.append(line)


def get_ranked_list(iterable, topics, extract_sentences=False):
    for k, grp in itertools.groupby(get_docs(iterable, extract_sentences=extract_sentences),
                                    lambda x: x['qid']):
        if k not in topics:
            print('Excluded topic {}'.format(k), file=sys.stderr)
            continue

        docs = list(grp)

        topic = topics[k].copy()
        topic['scores'] = {}
        topic['rels'] = {}
        for doc in docs:
            topic['scores'][doc['docno']] = doc['score']
            topic['rels'][doc['docno']] = doc['rel']
        yield {'topic': topic, 'docs': docs}


if __name__ == '__main__':
    class CustomHelpFormatter(argparse.HelpFormatter):
        def __init__(self, *args, **kwargs):
            kwargs['max_help_position'] = 40
            super(CustomHelpFormatter, self).__init__(*args, **kwargs)

    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    parser.add_argument('--exclude-qids', metavar='QIDS',
                        help='list of qids separated by whitespace')
    parser.add_argument('-s', '--extract-sentences', action='store_true', default=False,
                        help='extract sentences')
    parser.add_argument('topics_json')
    parser.add_argument('trectext')
    args = parser.parse_args()

    topics_data = json.load(smart_open(args.topics_json))
    topics = {q['number']: {'qid': q['number'], 'title': q['text']}
              for q in topics_data['queries']}

    if args.exclude_qids:
        for qid in args.exclude_qids.split(','):
            del topics[qid]

    ranked_list = tqdm(
        get_ranked_list(smart_open(args.trectext), topics, extract_sentences=args.extract_sentences),
        desc='Read topics')
    print(json.dumps(list(ranked_list), indent=2))
