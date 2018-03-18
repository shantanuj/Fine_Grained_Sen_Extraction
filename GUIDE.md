1) Load vocab as pickle file, and word vector path in config.py

2) If processed word embeddings do no not exist, then program will run trimming of word vectors from intersection of given vocab and vector path.

3) load train, test, etc in data/ folder. Specify path in model/config.py

IMP-> We assume that training/testing data has been properly processed, tokenized and tagged, etc. 
Certain normalizeation functions (only to convert to lowercase during testing are used).
_
4) Give model a name to save it later (restored for training)

5) For new vocab either use config or manually update vocab pickle file. 

6) Do not run make data
