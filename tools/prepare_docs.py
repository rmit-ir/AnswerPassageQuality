"""
Extract document texts from an index to accompany the input ranked list

The output is in trecweb format.
"""
from __future__ import print_function

import argparse
import itertools
import os.path
import re
import subprocess

from tqdm import tqdm


def dumpindex(repo, cmd, *args):
    p = subprocess.Popen(['dumpindex', repo, cmd] + list(args),
                         stdout=subprocess.PIPE)
    return p.communicate()[0]


def parse_trecweb(iterable):
    """Parse TRECWEB-format document content"""
    for line in iterable:
        if line.startswith('<DOCHDR>'):
            break
    dochdr = []
    for line in iterable:
        if line.startswith('</DOCHDR>'):
            break
        dochdr.append(line)
    return {'dochdr': dochdr, 'content': list(iterable)}


def parse_qrels(iterable):
    """Parse qrels entries"""
    for line in iterable:
        qid, _, docno, rel = line.split()
        yield {'qid': qid, 'docno': docno, 'rel': int(rel)}


def parse_trec_run(iterable):
    """Parse run entries"""
    for line in iterable:
        qid, _, docno, rank, sim, _ = line.split(None, 5)
        yield {'qid': qid, 'docno': docno, 'rank': int(rank), 'sim': float(sim)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-index', metavar='PATH',
                        help='load from Indri index')
    source.add_argument('--from-dir', metavar='PATH',
                        help='load from directory')
    parser.add_argument('-k', type=int,
                        help='process documents only up to rank K')
    parser.add_argument('qrels',
                        help='qrels file')
    parser.add_argument('run_file',
                        help='result file')
    parser.set_defaults(k=100)
    args = parser.parse_args()

    def _load_from_index(qid, docno):
        assert args.from_index
        docid = dumpindex(args.from_index, 'documentid', 'docno', docno).strip()

        if docid == '':
            return (
                '<DOC>\n'
                '<DOCNO>{}</DOCNO>\n'
                '</DOC>\n'.format(docno)
            )

        text = dumpindex(args.from_index, 'documenttext', docid).strip()

        if text.startswith('<DOC>') and text.endswith('</DOC>'):
            # TRECWEB format
            text = text.replace('<DOC>', '').replace('</DOC>', '')
            return parse_trecweb(iter(text.splitlines()))
        elif text.startswith('HTTP/'):
            # HTML (as part of HTTP response)
            text = text[text.find('\n\n') + 2:]
            text = text.replace('<DOC>', '').replace('</DOC>', '')
            docdata = dumpindex(args.from_index, 'documentdata', docid)
            m = re.search(r'^url: (\S+)$', docdata, flags=re.MULTILINE)
            url = m.group(1) if m is not None else ''
            return {'dochdr': [url], 'content': text.splitlines()}
        else:
            raise Exception('invalid data')


    def _load_from_dir(qid, docno):
        assert args.from_dir
        filepath = os.path.join(args.from_dir, qid, docno)
        trecweb = open(filepath, 'rU').read()
        trecweb = trecweb.replace('<DOC>', '')
        trecweb = trecweb.replace('</DOC>', '')
        return trecweb

    loader = None
    if args.from_index:
        loader = _load_from_index
    elif args.from_dir:
        loader = _load_from_dir

    # data_parser = parse_trecweb

    qrels = {(e['qid'], e['docno']): e['rel'] for e in parse_qrels(open(args.qrels))}

    tbar = tqdm(itertools.groupby(parse_trec_run(open(args.run_file)), lambda x: x['qid']))
    for qid, grp in tbar:
        thelist = list(grp) if args.k is None else [r for r in grp if r['rank'] <= args.k]
        for i, row in enumerate(thelist, 1):
            tbar.set_description('Process topic {} ({}/{})'.format(qid, i, len(thelist)))
            tbar.refresh()
            rel = qrels.get((qid, row['docno']), 0)
            # unparsed = loader(qid, row['docno'])
            # parsed = data_parser(iter(unparsed.splitlines()))
            parsed = loader(qid, row['docno'])

            print(
                '<DOC>\n'
                '<DOCNO>{docno}-{qid}</DOCNO>\n'
                '<QID>{qid}</QID>\n'
                '<SCORE>{score}</SCORE>\n'
                '<RELEVANCE>{rel}</RELEVANCE>\n'
                '<DOCHDR>\n'
                '<![CDATA[\n'
                '{dochdr}\n'
                ']]>\n'
                '</DOCHDR>\n'
                '{content}\n'
                '</DOC>'.format(dochdr='\n'.join(parsed['dochdr']),
                                content='\n'.join(parsed['content']),
                                qid=qid,
                                docno=row['docno'],
                                score=row['sim'],
                                rel=rel)
            )
