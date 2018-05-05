# Experiment: Gov2

This experiment is based on the TREC GOV2 dataset. To obtain the dataset, go to
the [TREC Web Research Collections] webpage hosted by University of Glasgow.

## Dependencies

Our scripts assume the following resources:

- `index`: symlink to the GOV2 indri index
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
- The CQA answers for TREC Topics 701-850 are made available in the file `ya.json.gz`.

[TREC Web Research Collections]: http://ir.dcs.gla.ac.uk/test_collections/
[wiki_en]: https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.zip
[RankLib]: https://www.lemurproject.org/ranklib.php
[trec_eval]: https://trec.nist.gov/trec_eval/
