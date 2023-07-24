# languagemodels

Shaun Harker

2023-07-04

## Installation and dataset preparation.

```bash
$ git clone https://github.com/shaunharker/languagemodels.git
$ cd languagemodels
$ mkdir data
$ cd data
$ wget https://the-eye.eu/public/AI/pile/train/01.jsonl.zst # takes a while
$ zstd -d 01.jsonl.zst
$ python3 preparedata.py
$ jupyter notebook
```

Now see the `2023-07-24-Experiment-1.ipynb` notebook to run the experiment.
