# Word2Vec: Custom implementation in PyTorch
Custom implementation of the original paper on Word2Vec - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

It uses minimum of third-party packages. Most of the functionality is implemented using basic features of PyTorch.

### Additional information:
* https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0
* https://muhark.github.io/python/ml/nlp/2021/10/21/word2vec-from-scratch.html
* https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html


## Overview
* There are 2 model architectures implemented in this project:
  - Continuous Bag-of-Words Model (CBOW), that predicts word based on its context
  - Continuous Skip-gram Model (Skip-Gram), that predicts context for a given word
* Models are trained on [text8](http://mattmahoney.net/dc/textdata.html) corpus which is the first 10<sup>9</sup> bytes of the English Wikipedia dump on Mar. 3, 2006
* Context for both models is represented as 5 words before and 5 words after the central word
* AdamW optimizer is used
* Trained for 5 epochs
* Results can be compared with reference [Gensim Word2Vec module](https://radimrehurek.com/gensim/models/word2vec.html) 

## Repository mirrors
- https://github.com/lacteolus/word2vec.git
- https://gitlab.atp-fivt.org/nlp2023/kosmachevdm-word2vec.git

## Repository structure
```
.
├── dataset
│   └── tesxt8.txt
├── notebooks
│   └── training.ipynb
├── results
│   ├── cbow
│   └── skipgram
├── src
│   ├── custom_word2vec.py
│   ├── dataloader.py
│   ├── gensim_word2vec.py
│   ├── helpers.py
│   ├── metric_monitor.py
│   ├── trainer.py
│   └── vocab.py
├── main.py
├── README.md
└── requirements.txt
```
- **dataset/text8.txt** - text8 corpus file
- **notebooks/training.ipynb** - demo for training procedure
- **results/** - folder for storing results
- **src/custom_word2vec.py** - custom Word2Vec model
- **src/dataloader.py** - dataloader related classes and functions
- **src/gensim_word2vec.py** - Gensim Word2Vec model
- **src/helpers.py** - helper functions
- **src/metric_monitor.py** - metric monitor class
- **src/vocab.py** - vocabulary class
- **main.py** - main script for training

## Usage
### Training in local environment

`python main.py`

Before running the command, the following parameters can be changed in `main.py` file:

- **MAX_VOCAB_SIZE** - Max vocabulary size
- **EPOCHS** - Number of epochs
- **MODEL_TYPE** - Model type to be used: "cbow" or "skipgram"
- **EMBEDDING_SIZE** - Embedding size
- **SAVE_PATH** - Path for saving results

By default, parameters are similar to ones used in Gensim.

### Using notebooks

## License
This project is licensed under the terms of the [MIT license](https://choosealicense.com/licenses/mit).