"""
Quality-biased ranking (Bendersky et al., 2011)
"""
import argparse
import bs4
import collections
import json
import math
# import re
import string

from smart_open import smart_open


# Module side
#
class Pipeline():
    """Feature extraction pipeline"""
    def __init__(self):
        self.jobs = []

    def add(self, features, adaptor=None):
        if not isinstance(features, (tuple, list)):
            features = [features]
        self.jobs.append({'adaptor': adaptor, 'extractors': features})

    def extract(self, item):
        vector = []
        for job in self.jobs:
            input_ = item if job['adaptor'] is None else job['adaptor'](item)
            for extractor in job['extractors']:
                vector.append(extractor(input_))
        return vector


PUNCTUATION_REMOVER = string.maketrans(string.punctuation, ' ' * len(string.punctuation))


def to_terms(text):
    return text.encode('utf8', errors='replace').translate(PUNCTUATION_REMOVER).split()


def UrlDepth(url):
    """The depth of the URL path"""
    pos = url.find('://')
    if pos >= 0:
        return url[pos+3:].count('/')
    else:
        return url.count('/')


def NumVisTerms(doc):
    """Number of visible terms on the page"""
    _, terms = doc
    return len(terms)


def NumTitleTerms(doc):
    """Number of terms in the page <title> field"""
    soup, _ = doc
    if soup.title is None:
        return 0
    else:
        return len(to_terms(soup.title.get_text()))


def AvgTermLen(doc):
    """Average length of visible term on the page"""
    _, terms = doc
    return float(sum(len(t) for t in terms)) / len(terms) if terms else 0


def FracAnchorText(doc):
    """Fraction of anchor text on the page"""
    soup, terms = doc
    terms_in_anchor_texts = sum(len(to_terms(tag.get_text())) for tag in soup.find_all('a'))
    return float(terms_in_anchor_texts) / len(terms) if terms else 0


def FracVisText(doc):
    """Fraction of visible text on the page"""
    soup, _ = doc
    try:
        pagesize = len(soup.decode_contents())
    except Exception:
        pagesize = 0
    return float(len(soup.get_text())) / pagesize if pagesize > 0 else 0


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


def FracTableText(doc):
    """Fraction of table text on the page"""
    soup, terms = doc
    terms_in_tables = 0
    for tag in soup.find_all('table'):
        if any(p.name == 'table' for p in tag.parents):
            continue
        terms_in_tables += len(to_terms(tag.get_text()))
    frac = float(terms_in_tables) / len(terms) if terms else 0
    assert frac <= 1
    return frac


# Data side
BLOCK_TAGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol', 'tr',
              'td', 'th', 'table', 'dl', 'dd', 'li', 'blockquote', 'pre',
              'address', 'title', 'head')


def SOUP_TERMS(doc):
    chunk = doc['text']
    soup = bs4.BeautifulSoup(chunk, 'lxml')
    for elem in soup(['br']):
        elem.insert_after('\n')
    for elem in soup(BLOCK_TAGS):
        elem.insert_after('\n')

    terms = to_terms(soup.get_text().lower())
    return soup, terms


def URL(doc):
    return doc['url']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus_json')
    parser.add_argument('stoplist')
    args = parser.parse_args()

    stoplist = set(l.strip() for l in smart_open(args.stoplist))

    pipeline = Pipeline()
    pipeline.add(UrlDepth, adaptor=URL)
    pipeline.add([NumVisTerms, NumTitleTerms, AvgTermLen, FracAnchorText, FracVisText,
                  Entropy, FracStops(stoplist), StopCover(stoplist), FracTableText], adaptor=SOUP_TERMS)

    ranked_lists = json.load(smart_open(args.corpus_json))
    for rl in ranked_lists:
        qid = rl['topic']['qid']
        for doc in rl['docs']:
            docno = doc['docno']
            rel = max(doc['rel'], 0)
            score = doc['score']
            vector = ' '.join(['{}:{}'.format(i, val) for i, val in enumerate(pipeline.extract(doc), 2)])

            print('{rel} qid:{qid} 1:{score} {vector} # {docno}'.format(**locals()))
