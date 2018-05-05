"""
Utilities for text processing
"""
from __future__ import print_function

import bs4
import os
import re
import sys

from stanford_corenlp_pywrapper import CoreNLP
from xml.sax.saxutils import escape


BLOCK_TAGS = ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol', 'tr',
              'td', 'th', 'table', 'dl', 'dd', 'li', 'blockquote', 'pre',
              'address', 'title', 'head')

HTML_TAG_PATTERN = re.compile(r'''
    < /?
    (?: a | abbr | acronym | address | applet | area | article | aside | audio
        | b | base | basefont | bdi | bdo | big | blockquote | body | br | button
        | canvas | caption | center | cite | code | col | colgroup | datalist | dd
        | del | details | dfn | dialog | dir | div | dl | dt | em | embed |
        fieldset | figcaption | figure | font | footer | form | frame | frameset |
        h1 | h2 | h3 | h4 | h5 | h6 | head | header | hr | html | i | iframe | img
        | input | ins | kbd | keygen | label | legend | li | link | main | map |
        mark | menu | menuitem | meta | meter | nav | noframes | noscript | object
        | ol | optgroup | option | output | p | param | pre | progress | q | rp |
        rt | ruby | s | samp | script | section | select | small | source | span |
        strike | strong | style | sub | summary | sup | table | tbody | td |
        textarea | tfoot | th | thead | time | title | tr | track | tt | u | ul |
        var | video | wbr )
    \b .*? > ''', flags=re.VERBOSE | re.IGNORECASE)

TRECWEB_TAG_PATTERN = re.compile(r'''
    < /? (?: doc | docno | text | headline ) \b .*? > ''', flags=re.VERBOSE | re.IGNORECASE)


def sanitize(chunk):
    """Remove HTML tags from the text content and leave space between paragraphs intact"""
    chunk = re.sub(r'<script\b.*?>.*?</script>', '', chunk, flags=re.DOTALL | re.IGNORECASE)
    chunk = re.sub(r'<style\b.*?>.*?</style>', '', chunk, flags=re.DOTALL | re.IGNORECASE)
    chunk = re.sub(r'<noscript>.*?</noscript>', '', chunk, flags=re.DOTALL | re.IGNORECASE)
    soup = bs4.BeautifulSoup(chunk, 'lxml')

    for elem in soup(['script', 'noscript', 'style']):
        elem.extract()
    for elem in soup(string=re.compile(r'<!--.*?-->', flags=re.DOTALL)):
        elem.extract()
    for elem in soup(['br']):
        elem.insert_after('\n')
    for elem in soup(BLOCK_TAGS):
        elem.insert_after('\n\n')

    text = soup.get_text()
    matches = re.findall(r'<.*?>', text)
    if matches:
        print(matches, file=sys.stderr)
        text = re.sub(HTML_TAG_PATTERN, ' ', text)
        text = re.sub(TRECWEB_TAG_PATTERN, ' ', text)
    return text


def get_paragraphs(text):
    buf = []
    for line in text.splitlines(True):
        if line.isspace():
            if buf:
                yield ''.join(buf)
                buf = []
            continue
        buf.append(line)
    if buf:
        yield ''.join(buf)


class SentenceDelimiter():
    def __init__(self, corenlp_path):
        self.proc = CoreNLP("ssplit", corenlp_jars=[os.path.join(corenlp_path, '*')])

    def get_sentences(self, text):
        res = self.proc.parse_doc(text)
        for sentence in res['sentences']:
            sentence_text = ' '.join(sentence['tokens']).encode('utf8')
            sentence_text = ' '.join(sentence_text.split())
            sentence_text = sentence_text.replace('-LRB-', '(').replace('-RRB-', ')')
            sentence_text = sentence_text.replace('-LSB-', '[').replace('-RSB-', ']')
            sentence_text = sentence_text.replace('-LCB-', '{').replace('-RCB-', '}')
            yield escape(sentence_text)
