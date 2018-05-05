# Experiment: CW09B

This experiment is based on the [ClueWeb09 dataset], the Category B subset (the first 50 million English webpages).

## Dependencies

Our scripts assume the following resources:

- `index`: symlink to the ClueWeb09B indri index
- `DrQA`: symlink to the DrQA repo
- `wiki.en.bin`: from the [pretrained fasttext embeddings][wiki_en]

## Get Started

To build the feature files for all experimental runs (may take a while):

    make

To run the experiment:

- Use python script `make_kfold_split.py` to partition data into 10 folds.
- Train a model for each fold with [RankLib] options `-ranker 4 -metric2t NDCG@20 -norm zscore`
- Make predictions and compile all results into a TREC run file.
- Evaluate the run using [trec_eval] and the `qrels` file.

To use the data:

- A set of TREC run files can be found in the directory `runs`.
- The CQA answers for TREC Web queries 1-200 are stored in the file `ya.json.gz`.

[ClueWeb09 dataset]: http://lemurproject.org/clueweb09/
[wiki_en]: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
[RankLib]: https://www.lemurproject.org/ranklib.php
[trec_eval]: https://trec.nist.gov/trec_eval/
