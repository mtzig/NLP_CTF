# NLP_CTF
Carleton College ML/NLP CS Comps 2023

Thomas Zeng, Teagan Johnson, Jared Chen, Nathan Hedgecock

A reimplementation of:

[Garg, Sahaj, et al. "Counterfactual fairness in text classification through robustness." Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society. 2019.](https://dl.acm.org/doi/pdf/10.1145/3306618.3317950)

## Installation

To install dependencies run the following command in the root directory:
```
pip install -r requirements.txt
```

We also require three files that must be manually installed due to file size.

1. Word2Vec embedings (specifically the GoogleNews-vecotors-negative300) to be installed in the `data` subdirectory
 ```
 wget -O GoogleNews-vectors-negative300.bin  'https://www.dropbox.com/s/mlg71vsawice3xd/GoogleNews-vectors-negative300.bin?dl=1'
 ```
2. glove embeddings (specifically the glove_840B_300d) to be installed in the `data` subdirectory -- only necessary if desire to run tests using glove instead of word2vec
```
wget -O glove_840B_300d.txt 'https://www.dropbox.com/s/a3meyi58v0jy4tv/glove_840B_300d.txt?dl=1'
```
3. Civil Comments dataset (We use a modified version from the paper [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://wilds.stanford.edu/)). Should be installed in the `data/civil_comments` subdirectory
```
wget -O civil_comments.csv 'https://www.dropbox.com/s/xv8zkmcmg74n0ak/civil_comments.csv?dl=1'
```

(NOTE: `wget` links may go stale as they are linked to my school address, if so you'll need to manually find them and install)



## Code Usage

For a quick demonstration of our experiments, open the `run.ipynb` jupyternotebook in the `notebooks` subdirectory. The notebook has an "Open in Colab" button at the top that allows one to run all the experiments in Colab (Note: A premium Colab instance may be required due to memory limitations of the free tier). 

To run our tests use `run.py`.

As an example the following group will run 10 trials of our baseline model and save results to `baseline_experiment.csv`.

```
python run.py baseline -v -n baseline_experiment
```

## Miscellaneous

`run.py` is currently configured to default to `mps` on Apple silicon device. Due to bugs in PyTorch implementation -- tests do not work on `mps`. We thus sugget manually setting the flag `-d cpu` if using `run.py` on an Apple silicon device.

## License

This source code is released under the MIT license, included [here](LICENSE).
