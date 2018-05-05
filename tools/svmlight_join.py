from __future__ import print_function

import argparse
import itertools
import re

from smart_open import smart_open
from tqdm import tqdm


def parse_svmlight_lines(iterable):
    for line in iterable:
        if line.startswith('#'):
            continue
        head, comment = line.split('#', 1)

        fields = head.split()
        rel = int(fields[0])
        qid = fields[1][4:]  # qid:XXXX
        vector = [float(f.split(':')[1]) for f in fields[2:]]

        docno = comment.strip()
        if docno.startswith('docno:'):
            docno = docno[6:]  # docno:XXXX
        yield {'qid': qid, 'rel': rel, 'vector': vector, 'docno': docno}


def parse_specs(specs):
    SPEC_PATTERN = re.compile(r'^(\d+)-(\d+)$')
    SPEC_ALL = ':all:'

    if len(specs) % 2 > 0:
        raise ValueError('missing filename or spec')

    filenames = specs[::2]
    fid_slices = []
    for spec in specs[1::2]:
        if spec == SPEC_ALL:
            fid_slices.append(slice(None))
        else:
            m = re.match(SPEC_PATTERN, spec)
            if m is None:
                raise ValueError("invalid spec '{}'".format(spec))
            a, b = int(m.group(1)), int(m.group(2))
            assert a <= b
            fid_slices.append(slice(a - 1, b))
    return filenames, fid_slices


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_args', nargs='+')
    args = parser.parse_args()

    try:
        vector_files, fid_slices = parse_specs(args.input_args)
    except Exception as e:
        parser.error(e)

    vector_size = None
    lines = itertools.izip(*[parse_svmlight_lines(smart_open(name)) for name in vector_files])
    for line in tqdm(lines, unit=''):
        sig = [(x['rel'], x['qid'], x['docno']) for x in line]
        assert sig.count(sig[0]) == len(sig)

        vector = list(itertools.chain(*[x['vector'][s] for x, s in zip(line, fid_slices)]))
        # vector = list(itertools.chain(*[x['vector'] for x in line]))
        if vector_size is None:
            vector_size = len(vector)
        assert len(vector) == vector_size

        rel, qid, docno = [line[0][k] for k in ('rel', 'qid', 'docno')]
        vector_data = ' '.join(['{}:{}'.format(i, v) for i, v in enumerate(vector, 1)])

        print('{rel} qid:{qid} {vector_data} # {docno}'.format(**locals()))
