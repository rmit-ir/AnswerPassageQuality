import argparse
import json
import requests
import time

from bs4 import BeautifulSoup
from tqdm import tqdm
from urlparse import urljoin


def get_search_result(query):
    base = 'https://au.answers.yahoo.com'
    r = requests.get('https://au.answers.yahoo.com/search/search_result',
                     params={'fr': 'uh3_answers_vert_gs', 'type': '2button', 'p': query})
    r.raise_for_status()

    soup = BeautifulSoup(r.text, 'lxml')
    return [{'url': urljoin(base, link['href']), 'title': link.get_text().strip()}
            for link in soup.select('ul#yan-questions li h3 a')]


def get_question_data(url):
    r = requests.get(url)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, 'lxml')

    html = {}
    title = unicode(soup.select('div#ya-question-detail h1')[0].get_text()).strip()

    body_span = soup.select('div#ya-question-detail span.ya-q-text')
    body_span += soup.select('div#ya-question-detail span.ya-q-full-text')
    body = unicode(body_span[-1].get_text()).strip()
    html['body'] = unicode(body_span[-1]).strip()

    answer_spans = soup.select('ul#ya-qn-answers span.ya-q-full-text')
    answers = [unicode(span.get_text()).strip() for span in answer_spans]
    html['answers'] = [unicode(span).strip() for span in answer_spans]

    best_answer_spans = soup.select('div#ya-best-answer span.ya-q-full-text')
    if best_answer_spans:
        best_answer = unicode(best_answer_spans[0].get_text()).strip()
        html['best_answer'] = unicode(best_answer).strip()
        answers.insert(0, best_answer)
        html['answers'].insert(0, html['best_answer'])
    else:
        best_answer = None
        html['best_answer'] = None

    return {'url': url,
            'title': title,
            'body': body,
            'best_answer': best_answer,
            'answers': answers,
            'html': html}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument('--stage1', action='store_true',
                       help='run stage 1: collect question-page URLS')
    stage.add_argument('--stage2', action='store_true',
                       help='run stage 2: parse and retrieve question/answer data')
    stage.add_argument('--stage3', action='store_true',
                       help='run stage 3: merge results')
    parser.add_argument('input_file')
    args = parser.parse_args()

    if args.stage1:
        data = json.load(open(args.input_file))
        queries = []
        for q in tqdm(data['queries'], 'Collect question-page URLs', unit='query'):
            queries.append({'qid': q['number'],
                            'text': q['text'],
                            'questions': list(get_search_result(q['text']))})
            time.sleep(2)
        print json.dumps(queries, indent=2)
    elif args.stage2:
        data = json.load(open(args.input_file))
        for q in tqdm(data, 'Gather best-matching question data', unit='query'):
            questions = []
            for url in [x['url'] for x in q['questions']]:
                questions.append(get_question_data(url))
            q['questions'] = questions
            print json.dumps(q)
    elif args.stage3:
        queries = []
        for line in open(args.input_file):
            queries.append(json.loads(line))
        print json.dumps(queries, indent=2)
