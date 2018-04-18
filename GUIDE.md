1) Load vocab as pickle file, and word vector path in config.py. Make sure the vocab matches with the stored embeddings. 

2) If processed word embeddings do no not exist, then program will run trimming of word vectors from intersection of given vocab and vector path (use trimmed vectors.py in data/Embeddings directory, use vocab as the one to be used) 

3) load train, test, etc in data/ folder. Specify path in model/config.py

IMP-> We assume that training/testing data has been properly processed, tokenized and tagged, etc. 
Certain normalizeation functions (only to convert to lowercase during testing are used).
_
4) Give model a name to save it later (restored for training)

NOTE: Train seq2seq with train_seq2seq.py. Once the loss reaches an optimum value, train absa with train_absa.py. Since training is separate but shares the same model, iterative training (seq2seq then absa then more seq2seq) is possible. 

REMEMBER TO change train_seq2seq in config.py to switch from seq2seq to absa training


5) For new vocab either use config or manually update vocab pickle file. 

