import argparse
import gzip
import os.path


class SVMLightFormatParser:
    def __init__(self, iterable):
        self.iterable = iterable
        self.line_generator = self._get_chunks()
        self.vector_generator = self._get_vectors()
        self.preamble = next(self.line_generator)

    def preamble(self):
        return self.preamble

    def lines(self):
        return self.line_generator

    def vectors(self):
        return self.vector_generator

    def _get_vectors(self):
        for line in self.line_generator:
            head, comment = line.strip().split('#', 1)
            components = head.split()
            rel, qid = int(components[0]), components[1][4:]
            yield {'rel': rel,
                   'qid': qid,
                   'fields': components[2:],
                   'comment': comment,
                   'line': line}

    def _get_chunks(self):
        preamble = []
        for line in self.iterable:
            if line.startswith('#'):
                preamble.append(line)
            else:
                yield preamble
                yield line
                break
        for line in self.iterable:
            yield line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int,
                        help='number of folds (default: %(default)s)')
    parser.add_argument('--no-validate', action='store_true',
                        help='do not use validate set')
    parser.add_argument('vector_file',
                        help='input vector file')
    parser.add_argument('output_path', nargs='?',
                        help='output path')
    parser.set_defaults(k=5)
    args = parser.parse_args()

    opener = gzip.open if args.vector_file.endswith('.gz') else file

    # the first pass through the data
    qids = []
    last_qid = None
    svmlight_parser = SVMLightFormatParser(opener(args.vector_file))
    for vector in svmlight_parser.vectors():
        if vector['qid'] != last_qid:
            qids.append(vector['qid'])
            last_qid = vector['qid']

    assert len(qids) == len(set(qids))

    subset_size = len(qids) / args.k
    subset = dict([(qid, min(args.k - 1, i / subset_size))
                   for i, qid in enumerate(qids)])

    # the second pass
    affix = os.path.basename(args.vector_file)
    if affix.endswith('.gz'):
        affix = affix[:-3]

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    training_outputs, validation_outputs, test_outputs = [], [], []
    for fold in range(args.k):
        training_outputs.append(
            file('{}/f{}.train.{}'.format(
                args.output_path, fold + 1, affix), 'wb'))
        if not args.no_validate:
            validation_outputs.append(
                file('{}/f{}.validation.{}'.format(
                    args.output_path, fold + 1, affix), 'wb'))
        test_outputs.append(
            file('{}/f{}.test.{}'.format(
                args.output_path, fold + 1, affix), 'wb'))

    target = [[] for x in range(args.k)]
    for fold in range(args.k):
        subset_list = range(args.k)
        subset_list = subset_list[fold:] + subset_list[:fold]

        target[subset_list[-1]].append(test_outputs[fold])
        if args.no_validate:
            for subset_number in subset_list[0:-1]:
                target[subset_number].append(training_outputs[fold])
        else:
            target[subset_list[-2]].append(validation_outputs[fold])
            for subset_number in subset_list[0:-2]:
                target[subset_number].append(training_outputs[fold])

    svmlight_parser = SVMLightFormatParser(opener(args.vector_file))
    for vector in svmlight_parser.vectors():
        subset_number = subset[vector['qid']]
        for fold in range(args.k):
            target[subset_number][fold].write(vector['line'])
