CONLL Dataset format and joint vocab (with special tokens-EOS, PAD, START included)  are the only things we require. 

From this word embeddings, graph embeddings are derived through relevant scripts. 

Some additional things to not:

1) Make sure vocab id is same across all domains, and contains PAD, EOS, NUM, and special tokens as shown in jupyter notebook. 

2) Make sure that the niters in data_utils.py matches with the domain, niter, word_id for graph embedding formation. niters is not increased for sentences with 0 length. 

3) Should we have different graph training based on domains. Or should we just train all together. Take extreme case. 

