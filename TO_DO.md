Stuff to try:
1) in current code-> move condense to after difference
2) Use a smaller seq2seq size and add word drop``
3) Allow for different embeddings being loaded, and domain identification--> just two different word embedding layers, and a placeholder number val/boolean val 
4) Try using uninitialized autoencoder-> if it performs similar then FORGET It

5) Graph embeddings

6) GRU 

7) Word archetypes/prototypes




1) 3rd Preprocess dependency structures, and incomplete dependenct structures. 
	a) Simple NLTK/Spacy script to store dependenct structure
	b) Assign node labels and graph labels-> store in graph2vec format
	c) Run graph2vec and get all possible embeddings. Store as vocab

2) MOST IMP: Make additional nnet transformer for learned weights and reduce hidden LSTM size to 300. 
	a) Just add a linear variable to compress the normal rep and missing rep through a transformation.  
	b) And handle the gradient flow
	c) Reduce LSTM dims to 300

3) 2nd Drop words for seq2seq. In training data have probability to remove words. 
	a)This might not be good, since a word the in context now may be removed just once, while in another context it is removed quite seldom. 
	b) The rep is now the actual meaning without the word. 
#	--> So that they don't have	

4) 5th Use GRU for autoencoding. Lesser parameters and only 1 encoded state.

4) 4th No time for Archetypal words- not right now. 
	a) Given trained model for absa, freeze it. 
	b) Have 6 distinct variables with variable properties. Minimize MDD or KLD of distributions of normal words and labels. 
	c) This can only happen once Tree Kernel is done.

5) 6th Reasoning structure formation
