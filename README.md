# AnswerPassageQuality

This repo contains code/scripts needed to reproduce the results in the SIGIR
'18 paper "Ranking Documents by Answer-Passage Quality".

If you use the data or code from this repo, please cite the following paper:
```
@inproceedings{yulianti_ranking_2018,
  author = {Yulianti, Evi and Chen, Ruey-Cheng and Scholer, Falk and
            Croft, W. Bruce and Sanderson, Mark},
  title = {Ranking Documents by Answer-Passage Quality},
  booktitle = {Proceedings of {SIGIR} '18},
  year = {2018},
  note = {to appear}
} 
```

## Get Started ##

Install the dependencies:

    pip install -r requirements.txt

Install the following packages from github repos:

* [stanford_corenlp_pywrapper](https://github.com/brendano/stanford_corenlp_pywrapper)
* [fastText](https://github.com/facebookresearch/fastText)
* [DrQA](https://github.com/facebookresearch/DrQA)

Note that, the DrQA package needs to be installed via `pip3` with all pretrained models downloaded:

    pip3 install .
    ./download.sh
    
## Ranking Experiments ##

* [GOV2](experiments/GOV2)
* [CW09B](experiments/CW09B)

## Contributors ##

* Ruey-Cheng Chen
* Evi Yulianti
