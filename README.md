# Fine Grained Sentiment Extraction

Implementation of bidirectional LSTM + missing word representations through seq2seq Autoencoders and dependency path embeddings. 

Further experiment code for domain adaptation on usage of different types of word embeddings (Glove, custom word2vec and geoDIST) is included. 


## Task

Aim: Develop domain adaptable classifier to label all aspect and opinion terms in a given sentence. Dataset used: Semeval dataset 2014 task 4 subtask1 -> Aspect and opinion extraction. 

```
I    love  Subway cookies but their sandwiches  are  so   dry.
OT    B-O   B-A    I-A    OT    OT     B-A      OT   B-O  I-O
```

## Read GUIDE.md for details on running code
Due to git size limits only glove 100d embeddings are provided. Drive links for other embeddings to be uploaded soon. 


## Coding style and framework of choice
Model code is developed on top of general sequence tagging in [Tensorflow framework by Guillaume Genthial repository](https://github.com/guillaumegenthial/sequence_tagging).
