BASE = $(realpath ../..)
RUN_FILE = runs/SDM


.DELETE_ON_ERROR:

.PHONY: runs

RUNS = vectors/SDM+PSG+BM25 vectors/SDM+PSG+QL vectors/SDM+PSG+EmbWiki \
       vectors/SDM+ILP+BM25 vectors/SDM+ILP+QL vectors/SDM+ILP+EmbWiki \
       vectors/SDM+DR vectors/SDM+DRDesc

runs: $(RUNS)


# Data
#
INDEX = index

$(INDEX):
	$(error '$@' should point to a GOV2 indri index)

freqstats.gz: $(INDEX)
	dumpindex $< v | awk '{ if ($$2 >= 5) print $$0 }' | gzip > $@

corpus.trecweb.gz: $(INDEX)
	python $(BASE)/tools/prepare_docs.py --from-index $< -k 100 qrels $(RUN_FILE) | gzip > $@

corpus.json.gz: corpus.trecweb.gz
	python $(BASE)/tools/jsonify.py -s --exclude-qids 703,769,816 topics.json $< | gzip > $@

wiki.en.bin:
	$(error '$@' can be downloaded from https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip)


# Baseline
#
vectors/QSDM: corpus.json.gz
	@mkdir -p vectors 
	stdbuf -o0 python $(BASE)/models/QSDM.py $< stoplist > $@

vectors/SDM+EE: corpus.json.gz ya.json.gz freqstats.gz
	@mkdir -p vectors
	python $(BASE)/models/EE.py --weights '0.3 0.7' --fb-original-weight 0.2 --fb-terms 60 \
	    corpus.json.gz ya.json.gz freqstats.gz > $@


# Experimental runs
#
vectors/SDM+PSG+BM25: corpus.json.gz ya.json.gz freqstats.gz
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py -avg-dl 100 \
	    -stoplist stoplist -freqstats freqstats.gz \
	    -dump-passages passages/PSG+BM25 PSG+BM25 corpus.json.gz ya.json.gz $@

vectors/SDM+PSG+QL: corpus.json.gz ya.json.gz freqstats.gz
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py -mu 100 \
	    -stoplist stoplist -freqstats freqstats.gz \
	    -dump-passages passages/PSG+QL PSG+QL corpus.json.gz ya.json.gz $@

vectors/SDM+PSG+EmbWiki: corpus.json.gz ya.json.gz wiki.en.bin
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py -k 10 -x0 0 \
	    -stoplist stoplist -fasttext-bin wiki.en.bin \
	    -dump-passages passages/PSG+EmbWiki PSG+Emb corpus.json.gz ya.json.gz $@

vectors/SDM+ILP+BM25: corpus.json.gz ya.json.gz freqstats.gz
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py -avg-dl 100 \
	    -stoplist stoplist -freqstats freqstats.gz \
	    -dump-passages passages/ILP+BM25 ILP+BM25 corpus.json.gz ya.json.gz $@

vectors/SDM+ILP+QL: corpus.json.gz ya.json.gz freqstats.gz
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py -mu 100 \
	    -stoplist stoplist -freqstats freqstats.gz \
	    -dump-passages passages/ILP+QL ILP+QL corpus.json.gz ya.json.gz $@

vectors/SDM+ILP+EmbWiki: corpus.json.gz ya.json.gz wiki.en.bin
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py -k 10 -x0 0 \
	    -stoplist stoplist -fasttext-bin wiki.en.bin \
	    -dump-passages passages/ILP+EmbWiki ILP+Emb corpus.json.gz ya.json.gz $@


# Ensemble runs
#
vectors/QSDM+EE+ILP+BM25: vectors/QSDM vectors/SDM+EE vectors/SDM+ILP+BM25
	python $(BASE)/tools/svmlight_join.py vectors/QSDM :all: vectors/SDM+EE 2-2 vectors/SDM+ILP+BM25 2-10 > $@


# DocumentReader
#
DRQA = DrQA

retriever.json: corpus.json.gz
	python $(BASE)/tools/make_passages.py $^ $@

doc.db: retriever.json
	python3 $(DRQA)/scripts/retriever/build_db.py $^ $@

doc-tfidf.npz: doc.db
	python3 $(DRQA)/scripts/retriever/build_tfidf.py $^ .
	mv doc-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz $@

DR.jsonl.gz: corpus.json.gz doc.db doc-tfidf.npz
	python3 $(BASE)/tools/run_DrQA.py \
	    --reader-model $(DRQA)/data/reader/multitask.mdl \
	    --retriever-model doc-tfidf.npz \
	    --doc-db doc.db --tokenizer simple \
	    corpus.json.gz $@

DRDesc.jsonl.gz: corpus.json.gz doc.db doc-tfidf.npz
	python3 $(BASE)/tools/run_DrQA.py \
	    --reader-model $(DRQA)/data/reader/multitask.mdl \
	    --retriever-model doc-tfidf.npz \
	    --doc-db doc.db --tokenizer simple \
	    --use-desc-topics trec_topics.json \
	    corpus.json.gz $@

vectors/SDM+DR: DR.jsonl.gz corpus.json.gz ya.json.gz
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py \
	    -stoplist stoplist -dr-jsonl DR.jsonl.gz \
	    -dump-passages passages/DR DR corpus.json.gz ya.json.gz $@

vectors/SDM+DRDesc: DRDesc.jsonl.gz corpus.json.gz ya.json.gz
	@mkdir -p vectors passages
	python $(BASE)/models/AnswerPassages.py \
	    -stoplist stoplist -dr-jsonl DRDesc.jsonl.gz \
	    -dump-passages passages/DRDesc DR corpus.json.gz ya.json.gz $@
