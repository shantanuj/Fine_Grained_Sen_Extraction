# Fine Grained Sentiment Extraction

Implementation of bidirectional LSTM + missing word representations through seq2seq Autoencoders. 

Further experiments for domain adaptation on usage of different types of word embeddings (Glove, custom word2vec and geoDIST), a 

Model code is developed on top of general sequence tagging in [Tensorflow framework by Guillaume Genthial repository](https://github.com/guillaumegenthial/sequence_tagging).

## Task

Given a sentence, give a tag to each word. A classical application is Named Entity Recognition (NER). Here is an example

```
I   love Subway cookies but their sandwiches are  so   dry.
O    B-O   B-A    I-A    O    O      B-A      O   B-O  I-O
```


