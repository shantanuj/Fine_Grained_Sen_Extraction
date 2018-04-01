'''
1) To do: Figure out line 201..
2) Add evaluation of seq2seq -> Get predictions,  see input word_ids and output word_Ids, just print that
3) Bridge as a function, cond decides what input to seq2seq is and edit the corresponding input to seq2seq

'''

import numpy as np
import os
import tensorflow as tf


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar
from .base_model import BaseModel
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}


    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence)
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")
	#to be used for seq2seq decoder
        if(self.config.use_seq2seq):
            self.decoder_targets = tf.placeholder(tf.int32, shape = [None, None], name="decoder_targets")
            self.max_sentence_length = tf.placeholder(dtype=tf.int32, shape=[], name="max_sentence_length")
            self.mask_matrix = tf.placeholder(dtype=tf.bool, shape=[None,None], name="mask_matrix")
            self.ones = tf.placeholder(dtype=tf.int32, shape =[None], name="ones")
            
        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")

	#Batch specific
	
	#self.pad_token = '<PAD>'
	#self.eos_token = '<END>'
	#self.PAD = self.config.vocab_words[self.pad_token]
 	#self.EOS = self.config.vocab_words[self.eos_token]

    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data:
        if self.config.use_chars:
            char_ids, word_ids = zip(*words)
            word_ids, sequence_lengths,max_sentence_length = pad_sequences(word_ids, 0)
            char_ids, word_lengths,_ = pad_sequences(char_ids, pad_tok=0,
                nlevels=2)
        else:
            word_ids, sequence_lengths,max_sentence_length = pad_sequences(words, 0)

        # build feed dictionary
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _, _ = pad_sequences(labels, self.config.vocab_tags['O'])
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout
        if (self.config.use_seq2seq):
            feed[self.decoder_targets] = word_ids
            if(not self.config.train_seq2seq):#Seq2seq has been trained-> Use for absa task
                np_mask_matrix = np.ones((max_sentence_length, max_sentence_length))
                
                a = np.array(range(max_sentence_length))
                np_mask_matrix[np.arange(len(a)),a] = 0
                
            else:
                np_mask_matrix = np.ones((1,1))
            np_mask_matrix = np_mask_matrix.astype(bool)
                
            feed[self.ones] = np.ones(shape=(len(words)), dtype="int32")
            feed[self.mask_matrix] =  np_mask_matrix
            feed[self.max_sentence_length] = max_sentence_length
            
            
	return feed, sequence_lengths

    
    
    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                self._word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                self._word_embeddings = tf.Variable(
                        self.config.embeddings,
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)

            word_embeddings = tf.nn.embedding_lookup(self._word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                        shape=[s[0]*s[1], s[-2], self.config.dim_char])
                word_lengths = tf.reshape(self.word_lengths, shape=[s[0]*s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                        shape=[s[0], s[1], 2*self.config.hidden_size_char])
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)

        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)
	#if(self.config.use_only_seq2seq):
	    #self.word_embeddings = tf.zeros([self.config.nwords, tf.shape(self.word_embeddings)[1],self.config.dim_word])
    
    def word_drop_pre_bridge(self, input_word_seq_tensor, seq_lengths, max_seq_length=None):
            #NOTE: There are no variables in the function, so shouldn't matter for the tf graph construction (hopefully)
        """ This function is only used during ABSA task. It takes in the input word sequence tensor and sequence lengths, and outputs a modified batch  where for each sentence, a row exists with normal sentence, and corresponding n rows with 1 missing word. 
        Same applies for the sequence lengths.

        In essence:
                For word_ids: Input 2d (n_batch,max_sequence_length) --> Intermediate 3d (n_batch, max_sequence_length, max_sequence_length) --> Output 2d (n_batch*max_sequence_length, max_sequence_length) --> Output 2d time major (max_sequence_length,n_batch*max_sequence_length)
            For seq_lens: Input 1d (n_batch,) --> Intermediate 2d (n_batch, max_sequence_length,) --> Output 1d (n_batch*max_sequence_length)
        """
        if self.config.use_seq2seq:
            if(max_seq_length is None):
                max_seq_length = self.max_sentence_length
            print(max_seq_length)
            #1) Create mask to select word indices ( 1 dropped every time)
            #np_mask_matrix = np.ones((max_seq_length, max_seq_length))
            #a = np.array(range(max_seq_length))
            #np_mask_matrix[np.arange(len(a)),a] = 0 #go through each row, and for that particular column set 0 (opposite of a diagonal matrix)
            #tf_mask_matrix = tf.convert_to_tensor(np_mask_matrix, dtype="bool")
            tf_mask_matrix = self.mask_matrix
            padding = tf.constant([[0,0],[0,1]],dtype="int32")

            #2nd operation add dimensions to both input tensors`
            resultant_tensor = tf.expand_dims(input_word_seq_tensor,0)
            tensor_seq_lengths = tf.expand_dims(seq_lengths, 0)

            #3rd operation--> Make tensor for seq_lengths for dropped indices (they're always 1 less)
            seq_lengths_for_dropped = tf.expand_dims(seq_lengths - self.ones,0)

            shape_input_word_seq = input_word_seq_tensor.get_shape()#tf.shape(input_word_seq_tensor)
            #4th operation --> looped Applying mask matrix to obtain dropped word ids; each result is appended to row of resultant_tensor
            drop_index = tf.constant(0)
    
            def condition(resultant_tensor, tensor_seq_lengths, drop_index):
                return drop_index<max_seq_length

            def body(resultant_tensor, tensor_seq_lengths, drop_index):
                f = lambda word_seq: tf.boolean_mask(word_seq, tf_mask_matrix[:,drop_index])#,padding,"CONSTANT") 
                '''Apply masking and padding'''
                temp = tf.pad(tf.map_fn(f, input_word_seq_tensor), padding, "CONSTANT") #We apply mask and then pad result to enable concatenation with original 
                temp = tf.expand_dims(temp,0)
                resultant_tensor = tf.concat([resultant_tensor,temp],0)
                tensor_seq_lengths = tf.concat([tensor_seq_lengths, seq_lengths_for_dropped],0)
                drop_index+=1
                return resultant_tensor, tensor_seq_lengths, drop_index    
	
    
            resultant_tensor, tensor_seq_lengths, drop_index = tf.while_loop(condition, body,[resultant_tensor, tensor_seq_lengths,0], shape_invariants=[tf.TensorShape([None,shape_input_word_seq[0],shape_input_word_seq[1]]),tf.TensorShape([None,shape_input_word_seq[0],]),drop_index.get_shape()])

	    #5th operation-> reshape of tensorshape_input_word_seq
            # The tensor is shaped such that the first n rows correspond to the first sentence (n is the sequence length)
            resultant_tensor_shape = tf.shape(resultant_tensor)
            resultant_tensor = tf.reshape(resultant_tensor, [resultant_tensor_shape[0]*resultant_tensor_shape[1], resultant_tensor_shape[2]])
            tensor_seq_lengths = tf.reshape(tensor_seq_lengths, [resultant_tensor_shape[0]*resultant_tensor_shape[1],])
            
	#NOTE: Converting the tensor from batch*time -> time*batch BECAUSE OUR SPECIFIC ENCODER expects in that manner
            resultant_tensor = tf.transpose(resultant_tensor, [1,0]) 
            return resultant_tensor, tensor_seq_lengths

   
	    
	     			
    def convert_tensors(self):
	#NOTE Word ids during training of seq2seq are of different format(time*batch) whereas in absa task they are fed normal as batch*time) 
	if(self.config.train_seq2seq):
	    self.seq2seq_input_sequences, self.seq2seq_input_sequence_lengths =  self.word_ids, self.sequence_lengths
        else:
	    self.seq2seq_input_sequences, self.seq2seq_input_sequence_lengths = self.word_drop_pre_bridge(self.word_ids, self.sequence_lengths)

    def add_seq2seq(self):
	"""This stores the seq2seq model which will be imported as part of the training graph since other options of creating a separate training graph/session and importing seemed lengthy. 

	1) It is to be first trained for autoencoding separately 
	2) Once the training is complete, user has to update config in model_config.py
	3) For usage in ABSA, the variable tf_encoded_concat_rep is used. It is updated to not trainable by blocking the gradient flow
"""
#NOTE: 1) There might be a more efficient manner to load and train the seq2seq separately, and then just use the final weights. 
#      2) Blocking gradients should not impact elements linked to this
        if(self.config.use_seq2seq):
	    
            with tf.variable_scope('seq2seq_encoder',reuse=tf.AUTO_REUSE):
                self.seq2seq_input_sequences_embeds = tf.nn.embedding_lookup(self._word_embeddings, self.seq2seq_input_sequences, name="word_embeddings")

                encoder_cell = LSTMCell(self.config.seq2seq_enc_hidden_size)

                ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell, cell_bw=encoder_cell, inputs = self.seq2seq_input_sequences_embeds, sequence_length = self.seq2seq_input_sequence_lengths, dtype = tf.float32, time_major=True))
               #encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs),2)
                encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c),1)
                encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h),1)

                self.encoder_final_state = LSTMStateTuple(c= encoder_final_state_c, h=encoder_final_state_h)
		if(not self.config.use_only_h):
               	    self.encoder_concat_rep = tf.concat([encoder_final_state_c, encoder_final_state_h], 1)	       
           	else:
		    c_shape = tf.shape(encoder_final_state_c)
		    self.encoder_concat_rep = tf.concat([encoder_final_state_h, tf.zeros([c_shape[0],c_shape[1]])],1)
	         #print(type(self.encoder_concat_rep))
		 #NOTE: Need to stop gradient flow once trained
                if(self.config.seq2seq_trained):
                    self.encoder_concat_rep = tf.stop_gradient(self.encoder_concat_rep) 
    
        
            with tf.variable_scope('seq2seq_decoder',reuse=tf.AUTO_REUSE):
                encoder_max_time, batch_size = tf.unstack(tf.shape(self.seq2seq_input_sequences))
         	#self.encoder_max = encoder_max_time
		#self.batch_max = batch_size 
                decoder_cell = LSTMCell(self.config.seq2seq_dec_hidden_size)

                decoder_lengths = self.sequence_lengths + 3 #2 additional terms
                W_dec = tf.Variable(tf.random_uniform([self.config.seq2seq_dec_hidden_size, self.config.nwords],-1,1), dtype = tf.float32)
                b_dec = tf.Variable(tf.zeros([self.config.nwords]), dtype = tf.float32)
   
                eos_time_slice = self.config.EOS*tf.ones([batch_size], dtype=tf.int32, name = "EOS")
                pad_time_slice = self.config.PAD*tf.ones([batch_size], dtype = tf.int32, name="PAD")

                eos_step_embedded = tf.nn.embedding_lookup(self._word_embeddings, eos_time_slice)
                pad_step_embedded = tf.nn.embedding_lookup(self._word_embeddings, pad_time_slice)
            
            def loop_fn_initial():
                initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
                initial_input = eos_step_embedded
                initial_cell_state = self.encoder_final_state
                initial_cell_output = None
                initial_loop_state = None  # we don't need to pass any additional information
                return (initial_elements_finished,initial_input,initial_cell_state,initial_cell_output,initial_loop_state)

            def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
                def get_next_input():
                    output_logits = tf.add(tf.matmul(previous_output, W_dec), b_dec)
                    prediction = tf.argmax(output_logits, axis=1)
                    next_input = tf.nn.embedding_lookup(self._word_embeddings, prediction)
                    return next_input

                elements_finished = (time >= decoder_lengths)
                
                finished = tf.reduce_all(elements_finished) # -> boolean scalar
                input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
                state = previous_state
                output = previous_output
                loop_state = None
                
                return (elements_finished, input, state, output, loop_state)

            def loop_fn(time, previous_output, previous_state, previous_loop_state):
                if previous_state is None:    # time == 0
                    assert previous_output is None and previous_state is None
                    return loop_fn_initial()
                else:
                    return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)
            with tf.variable_scope("seq2seq_decoding"):
                decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
                decoder_outputs = decoder_outputs_ta.stack()
            
                decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
                decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
                decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W_dec), b_dec)
                self.decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.config.nwords))
                self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
      
    #If training of seq2seq is to be done, then we need to add loss and cost function to the graph
       # if(self.config.train_seq2seq and self.config.use_seq2seq):
        #    with tf.variable_scope('seq2seq_training'):
         #       stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(decoder_targets, depth = self.config.nwords, dtype = tf.float32), logits = self.decoder_logits,)
          #      self.seq2seq_loss = tf.reduce_mean(stepwise_cross_entropy)
            
    def condense_layer(self,concat_rep,dim1=0,dim2=0):
	if(self.config.use_seq2seq and self.config.use_condense_layer): 
	    with tf.variable_scope("condense", reuse=tf.AUTO_REUSE):
		W_con = tf.get_variable("W_con", dtype =tf.float32, shape = [self.config.seq2seq_enc_hidden_size*4, self.config.condense_dims]) 
		b_con = tf.get_variable('b_con', dtype = tf.float32, shape= [self.config.condense_dims], initializer = tf.zeros_initializer())
		#NOTE: Have to experiment with activation function here
	        if(concat_rep is not None):
		    return tf.matmul(concat_rep,W_con)+b_con #tf.nn.relu(tf.matmul(concat_rep,W_con)+b_con)
	    	else:
		    return tf.zeros([dim1,dim2,self.config.condense_dims])

    def bridge_seq2seq_embeddings(self):
	#This is to convert the seq2seq outputs of shape 2d Time*Batch --> Perform comparison of missing word with all words-->convert into 3d shape(Batch*Time*Embeds), and then concatenate as batch*Time*Embeds with word_embeddings
        if self.config.use_seq2seq:
	    #print(self.word
	    dim1 = tf.shape(self.word_ids)[-1]+1 #Extra since normal rep + n missing reps
	    dim2 = tf.shape(self.word_ids)[0]
	    #print(dim1, type(dim1))
	    #print(type(self.encoder_concat_rep))
            seq2seq_encoder_out = tf.reshape(self.encoder_concat_rep,[dim1, dim2, self.config.seq2seq_enc_hidden_size*4]) #Reshape time major,dims to Time(0th is normal one)*Batch*Dims output

	    #NOTE: seq2seq_encoder_out[0:]<-- corresponds to normal representation for all sentences. The 2nd dimension is the batch, the first dimension is seq_length+1
	    #So we have to perform an operation on the first dimension first value(normal all words rep of sentence) with all other missing word reps. 
            if(self.config.use_condense_layer):
		#concat_within = lambda row: tf.concat([row,seq2seq_encoder_out[0,:]],1)
	#	self.shape_before = tf.shape(seq2seq_encoder_out)
		#seq2seq_encoder_out = tf.map_fn(concat_within,seq2seq_encoder_out,dtype=tf.float32)#[1:,:]
		self.seq2seq_encoder_embeds = tf.map_fn(self.condense_layer, seq2seq_encoder_out)#[1:,:]
		self.seq2seq_encoder_embeds = tf.subtract(self.seq2seq_encoder_embeds, self.seq2seq_encoder_embeds[0,:])[1:,:] #tf.abs 

		#self.shape_inter = tf.shape(seq2seq_encoder_out)
		#seq2seq_encoder_out = tf.reshape(seq2seq_encoder_out, [(dim1)*dim2, 2*self.config.seq2seq_enc_hidden_size*4])
		#seq2seq_encoder_out = self.condense_layer(seq2seq_encoder_out)
		#seq2seq_encoder_out = tf.reshape(seq2seq_encoder_out, [(dim1), dim2,self.config.condense_dims])
		#self.seq2seq_encoder_embeds = seq2seq_encoder_out[1:,:] 
	#	self.shape_after = tf.shape(self.seq2seq_encoder_embeds)
	        #self.seq2seq_encoder_embeds = tf.map_fn(self.condense_layer, seq2seq_encoder_out)
	    
		#1) Concatenate 0th with every element downwards
		#2) Perform batch computation of each row and then column
	    else:    
		self.seq2seq_encoder_embeds = tf.subtract(seq2seq_encoder_out, seq2seq_encoder_out[0,:])[1:,:]  #tf.abs
		  #NOTE#NOTE#NOTE#NOTE Have to replace with a generic function-subtract, KL, MMD
            self.seq2seq_encoder_embeds = tf.transpose(self.seq2seq_encoder_embeds, perm =[1,0,2])
 	    
	    if(self.config.use_cosine_sim):
	    	def get_cosine_similarity_within(tensor):
		    res_tensor = tf.nn.l2_normalize(tensor,dim=2)
		    res_tensor = tf.reduce_sum(tf.multiply(res_tensor, res_tensor[0,:,])[1:],2,keep_dims=True)
		   
                    res_tensor = tf.transpose(res_tensor, perm = [1,0,2])
		    return res_tensor
	        cos_sim_h = get_cosine_similarity_within(seq2seq_encoder_out[:,:,:self.config.seq2seq_enc_hidden_size*2])
		cos_sim_c = get_cosine_similarity_within(seq2seq_encoder_out[:,:,self.config.seq2seq_enc_hidden_size*2:])
		self.seq2seq_encoder_cosine_similarities = tf.concat([cos_sim_h, cos_sim_c], axis =-1)
#	    	normalized_seq2seq_enc = tf.nn.l2_normalize(seq2seq_encoder_out, dim=2)
#	    	self.seq2seq_encoder_cosine_similarities = tf.reduce_sum(tf.multiply(normalized_seq2seq_enc, normalized_seq2seq_enc[0,:,])[1:], 2, keep_dims=True)
	    	#self.seq2seq_encoder_cosine_similarities = tf.transpose(self.seq2seq_encoder_cosine_similarities, perm =[1,0,2])
		if(self.config.use_only_cosine_sim):
		    self.seq2seq_encoder_embeds = tf.concat([self.seq2seq_encoder_cosine_similarities,tf.zeros([dim2,dim1-1,self.config.seq2seq_enc_hidden_size*4-2])], axis=-1)
		    #self.seq2seq_encoder_embeds = tf.concat([self.seq2seq_encoder_cosine_similarities,tf.zeros([dim2,dim1-1,self.config.seq2seq_enc_hidden_size*4])], axis=-1)	 
		else:
		    self.seq2seq_encoder_embeds = tf.concat([self.seq2seq_encoder_embeds[:,:,:-2], self.seq2seq_encoder_cosine_similarities], axis=-1)   #NOTE NOTE NOTE NOTE NOTE We drop the last rep dim to allow for cosine to be incorporated(static graph)
	    
            if(self.config.train_seq2seq):
		#NOTE NOTE NOTE NOTE : This is done to allow training optimization of absa to be added to graph (else it links word embeddings) #NOTE: ALSO, this only works when we take enc h+c bidirectional rep (multiply by 4)
		if(self.config.use_condense_layer):
		    condense_dummy = self.condense_layer(tf.zeros([dim2*(dim1-1),self.config.seq2seq_enc_hidden_size*4]))
		    self.word_embeddings = tf.concat([self.word_embeddings, tf.reshape(condense_dummy, [dim2, dim1-1,self.config.condense_dims])], axis=-1)#self.condense_layer(None,dim2,dim1-1)], axis =-1) 
		else:
		    self.word_embeddings = tf.concat([self.word_embeddings, tf.zeros([dim2, dim1-1,self.config.seq2seq_enc_hidden_size*4])], axis =-1)
	    else:
		#self.word_embeddings = tf.concat([self.word_embeddings, tf.zeros([dim2, dim1-1,self.config.seq2seq_enc_hidden_size*4])], axis =-1)
		if(self.config.use_only_seq2seq):
		    self.word_embeddings = tf.concat([tf.zeros([dim2,dim1-1,self.config.dim_word]),self.seq2seq_encoder_embeds],axis=-1)
		else:
		    self.word_embeddings = tf.concat([self.word_embeddings, self.seq2seq_encoder_embeds], axis =-1)
		  	    
	    
    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            self.lstm_out_shape = tf.shape(self.word_embeddings)
	    output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])
	    #if(self.config.train_seq2seq and self.config.use_seq2seq):
	#	self.logits = tf.stop_gradient(self.logits)

    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
        #if self.config.use_crf and not self.config.train_seq2seq:
            
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.loss = tf.reduce_mean(-log_likelihood)
        else: #Use when no crf and se
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            mask = tf.sequence_mask(self.sequence_lengths)
            losses = tf.boolean_mask(losses, mask)
            self.loss = tf.reduce_mean(losses)

        if(self.config.use_seq2seq):
            
            self.stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.decoder_targets, depth = self.config.nwords, dtype = tf.float32), logits = self.decoder_logits,)
            self.seq2seq_loss = tf.reduce_mean(self.stepwise_cross_entropy)
        
        # for tensorboard
        if(self.config.train_seq2seq and self.config.use_seq2seq):
            tf.summary.scalar("seq2seq_loss",self.seq2seq_loss)
        else:
            tf.summary.scalar("loss", self.loss)
        

    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        if(not self.config.use_seq2seq):
            self.add_logits_op()
            self.add_pred_op()
            self.add_loss_op()
        else: 
	    self.convert_tensors()
	    
            self.add_seq2seq()
            self.bridge_seq2seq_embeddings()
	    self.add_logits_op()
	    self.add_pred_op()
            self.add_loss_op()		
       
        #self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        if(self.config.use_seq2seq): #and self.config.train_seq2seq):#This is also a node in the graph and hence needs to be stored
        # Generic functions that add training op and initialize session
	    # self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
	     self.add_train_op(self.config.lr_method, self.lr, self.seq2seq_loss, self.config.clip, True)
        #else:
	 #    self.add_train_op(self.config.lr_method, self.lr, self.seq2seq_loss, self.config.clip, True)
 	  #   self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip) 
        
        self.add_train_op(self.config.lr_method, self.lr, self.loss, self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # iterate over the sentences because no batching in vitervi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch_seq2seq(self, train, dev, epoch):
        """for seq2seq training"""
        """train is a list of sequences"""
        """dev is also a list of sequences"""
        
        batch_size = self.config.seq2seq_batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)
        #train_op = tf.train.AdamOptimizer(learning_rate= self.config.lr).minimize(self.seq2seq_loss)
        tr_ep_loss = 0.0
        #train_batch_generator = self.gen_batch_seq2seq(train,batch_size)
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            #print("TR",len(words),len(words[0]), len(labels), len(labels[0]))
            df = self.next_feed(words, lr=self.config.lr, dropout = self.config.dropout_seq2seq)
            #if(self.config.complete_autoencode_including_test):
                #dev_batch = self.next_feed(words,lr=self.config.lr, dropout = self.config.dropout_seq2seq)
            	#df = self.merge_feeds(df,dev_batch)
	#  cross_entropy, decoder_logits, encoder_useful_state = self.sess.run([self.stepwise_cross_entropy, self.decoder_logits,self.encoder_concat_rep], feed_dict =df)
            _, train_loss, summary = self.sess.run([self.seq2seq_train_op, self.seq2seq_loss, self.merged], feed_dict = df)
            #print("lstm out",lstm_out_shape)
	   #print("ENC_TIME",enc_time)
           # print("BATCH_SIZE",b_size) 
            #print(cross_entropy)
            #print(decoder_logits[0])
            #print(encoder_useful_state[0])
    	    prog.update(i + 1, [("train loss", train_loss)])
	    tr_ep_loss+=train_loss
       	    #if(i%70)
	     #   print("ACTUAL,PREDICTED",words[0],predictions[0])	
        for words, labels in minibatches(dev, 400):
            
            #te_loss = 5
            #predictions, encoder_useful_state = self.sess.run([self.decoder_prediction, self.encoder_concat_rep], dev_batch)
	    #te_loss, predictions = self.sess.run([self.seq2seq_loss,self.decoder_prediction], feed_dict = dev_batch)
	    if(self.config.complete_autoencode_including_test):
		 dev_batch = self.next_feed(words,lr=self.config.lr, dropout = self.config.dropout_seq2seq)		       
		 _,te_loss, predictions = self.sess.run([self.seq2seq_train_op,self.seq2seq_loss,self.decoder_prediction], feed_dict = dev_batch)
	    else:
	    	dev_batch = self.feed_enc(words)
	    	te_loss, predictions = self.sess.run([self.seq2seq_loss,self.decoder_prediction], feed_dict = dev_batch)
	    tr_ep_loss +=te_loss

	    #te_loss = self.compute_seq2seq_acc(words, predictions)
	    #prog.update(i + 1, [("test loss", te_acc)])
            #print("TE",len(words),len(words[0]),len(predictions), len(predictions[0]))
        #print("Word embedding shape", word_embeds.shape)
        #print("Word embeds for 0th sentence and 1st word",word_embeds[0][1])
        words = np.transpose(np.array(words))
        predictions = np.transpose(predictions)
	
        print("ACTUAL,PREDICTED", words[1], predictions[1])
	#print("AC, PR", words[2],predictions[2])
        #print("Encoder state 0: {}".format(encoder_useful_state[0][0]))
        msg = "Autoencoding testing loss: {}".format(te_loss)
	if(self.config.complete_autoencode_including_test):
	    model_comparison_val =  tr_ep_loss/i
	else:
	    model_comparison_val = te_loss
        #te_loss = 5
	self.logger.info(msg)
	#print("Cumulative loss: {}".format(float(train_loss)+float(te_loss)))
        return model_comparison_val

    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr, self.config.dropout)
	    
	    #shape_after, shape_before = self.sess.run([self.shape_after, self.shape_before], feed_dict = fd)
	    #print(shape_after,  shape_before)
	    _, train_loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict = fd)
            #enc_rep, _, train_loss, summary = self.sess.run(
           #         [self.encoder_concat_rep,self.train_op, self.loss, self.merged], feed_dict=fd)
	    
            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)
	#print("W_embeds 0", w_embeds.shape)
	#print(w_embeds[0][0])
        if(self.config.use_seq2seq):
            for words, labels in minibatches(dev, self.config.batch_size):
                dev_batch = self.feed_enc(words)
           	te_loss = 5
            	word_embeds, encoder_useful_state = self.sess.run([self.word_embeddings,self.encoder_concat_rep], dev_batch)
            #print("Word embedding shape", word_embeds.shape)
        #print("Word embeds for 0th sentence and 1st word",word_embeds[0][1]) 
        #print("Encoder state 0: {}".format(encoder_useful_state[0][0]))
        #print(len(encoder_useful_state[0][0]))
	
        metrics = self.run_evaluate(dev)
        msg = " - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]

    def calculate_f1(self, tp,fp,tn,fn):
        if(tp+fn==0):
            recall = 0
        else:
            recall = float(tp)/(tp+fn)
        if(tp+fp ==0):
            precision = 0
        else:
            precision = float(tp)/(tp+fp)
        if(precision+recall==0):
            f1 = 0
        else:
            f1 = 2*(precision*recall)/(precision+recall)
        return f1, recall, precision

    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        asp_tp = 0.
        asp_fp = 0.
        asp_tn = 0.
        asp_fn = 0.
    
        op_tp = 0.
        op_fp = 0.
        op_tn = 0.
        op_fn = 0.
    
        ot_tp = 0.
        ot_fp = 0.
        ot_tn = 0.
        ot_fn = 0.
        
        tag2id = self.config.vocab_tags 
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                
                
                for actual,pred in zip(lab, lab_pred):
                    actual = actual
                    pred = pred
                    #print(type(tag2id['B-A']), type(actual), type(pred))
                    #print(actual, actual ==4)
                    #print(pred, pred ==4)
                    if(actual == tag2id['B-A'] or actual == tag2id['I-A']): #BA or IA-> Replace by tag2id later --> 0 and 2 for i-a and B-a; 1 and 3; 4
                        if(pred == tag2id['B-A'] or pred == tag2id['I-A']):
                            asp_tp +=1
                            op_tn +=1
                            ot_tn +=1
                        else:
                            if(pred==tag2id['B-O'] or pred==tag2id['I-O']): 
                                asp_fn+=1
                                op_fp+=1
                                ot_tn+=1
                            elif(pred==tag2id['O']):
                                asp_fn+=1
                                ot_fp+=1
                                op_tn+=1
                            else:
                                print("Somethings wrong in prediction")
                            
                    elif(actual==tag2id['B-O'] or actual==tag2id['I-O']): #BO or IO
                        if(pred==tag2id['B-O'] or pred==tag2id['I-O']):
                            op_tp +=1
                            asp_tn +=1
                            ot_tn +=1
                        else:
                            if(pred == tag2id['B-A'] or pred==tag2id['I-A']): 
                                op_fn+=1
                                asp_fp+=1
                                ot_tn+=1
                            elif(pred==tag2id['O']):
                                op_fn+=1
                                ot_fp+=1
                                asp_tn+=1
                            else:
                                print("Somethings wrong in prediction")
                                
                                
                    elif(actual == tag2id['O']):
                        if(pred==tag2id['O']):
                            ot_tp +=1
                            asp_tn +=1
                            op_tn +=1
                        else:
                            if(pred == tag2id['B-A'] or pred==tag2id['I-A']): 
                                ot_fn+=1
                                asp_fp+=1
                                op_tn+=1
                            elif(pred==tag2id['B-O'] or pred==tag2id['I-O']):
                                ot_fn+=1
                                op_fp+=1
                                asp_tn+=1
                            else:
                                print("Somethings wrong in prediction")                                
                    else:
                        print("Somethings wrong")
                   
                                
                                
                                
                
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred,
                                                 self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)
        
        assert(asp_tp+asp_fp+asp_tn+asp_fn == op_tp+op_fp+op_tn+op_fn == ot_tp+ot_fp+ot_tn+ot_fn)
        #print(tag2id)
        #print(1)
        #print(asp_tp, asp_fp, asp_tn, asp_fn)
        asp_scores = self.calculate_f1(asp_tp,asp_fp,asp_tn,asp_fn)
        #print(2)
        op_scores = self.calculate_f1(op_tp,op_fp,op_tn,op_fn)
        #print(3)
        ot_scores = self.calculate_f1(ot_tp,ot_fp,ot_tn,ot_fn)
        
                
                
                
        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1, "asp_f1":100*asp_scores[0], "op_f1":100*op_scores[0], "ot_f1":100*ot_scores[0]}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        preds = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return preds


    def gen_batch_seq2seq(self, idd_data, batch_size):
        #np.random.shuffle(idd_data)
        #batch_size = self.config.seq2seq_batch_size
        print("Batch size", batch_size)
        rem = len(idd_data)%(self.config.seq2seq_batch_size)
        num_batches = (len(idd_data)/self.config.seq2seq_batch_size)
        if(rem>0):
            num_batches = num_batches+1
        print(idd_data)
        for i in range(num_batches):
            if(i==num_batches -1 and (not rem==0)):
                yield(idd_data[i*batch_size:])
            else:
                yield(idd_data[i*batch_size:(i+1)*batch_size])
    
                             
    def batch_modify(self, inputs, max_sequence_length=None):
        """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
        """
    
        sequence_lengths = [len(seq) for seq in inputs]
        batch_size = len(inputs)
    
        if max_sequence_length is None:
            max_sequence_length = max(sequence_lengths)
    
        inputs_batch_major = self.config.PAD*np.ones(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
        inputs_time_major = inputs_batch_major.swapaxes(0, 1)

        return inputs_time_major, sequence_lengths
                             
    def next_feed(self, batch, lr = 0.02, labels = None, dropout= 1.0):
        encoder_inputs_, encoder_input_lengths_ = self.batch_modify(batch)
        #print(self.config.EOS, self.config.PAD)
        max_sentence_length = encoder_inputs_.shape[0] #Time major
        batch_size = encoder_inputs_.shape[-1] #Time*Batch
        decoder_targets_, _ = self.batch_modify(
            [(sequence) + [self.config.EOS] + [self.config.PAD] * 2 for sequence in batch] #additional 3 spaces
        )
        feed = {
            self.word_ids: encoder_inputs_,
            self.sequence_lengths: encoder_input_lengths_,
            self.decoder_targets: decoder_targets_,
        }
#	print("HELLO")
#	print(encoder_inputs_[0])
#	print(decoder_targets_[0])
        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
	
        feed[self.labels] = decoder_targets_
 
        if labels is not None:
            labels, _ = pad_sequences(labels, self.config.vocab_tags['O'])
            feed[self.labels] = labels
	
        if lr is not None:
            feed[self.lr] = self.config.lr

        if dropout is not None:
            feed[self.dropout] = dropout
        if (self.config.use_seq2seq):
            
            if(not self.config.train_seq2seq):#Seq2seq has been trained-> Use for absa task
                np_mask_matrix = np.ones((max_sentence_length, max_sentence_length))
                feed[self.decoder_targets] = encoder_inputs_
                a = np.array(range(max_sentence_length))
                np_mask_matrix[np.arange(len(a)),a] = 0
                #NOTE: When for ABSA task we feed encoder, we feed as normal (the tf graph converts to time major format)
                feed[self.word_ids] = encoder_inputs_.swapaxes(0,1)
                
            else:
                np_mask_matrix = np.ones((1,1))
            
	    np_mask_matrix = np_mask_matrix.astype(bool)
                
            feed[self.ones] = np.ones(shape=(batch_size), dtype="int32")
            feed[self.mask_matrix] =  np_mask_matrix
            feed[self.max_sentence_length] = max_sentence_length
        return feed

    def feed_enc(self, enc_batch, lr = 0.02, labels = None, dropout= 1.0):
    
        
        encoder_inputs_, encoder_input_lengths_ = self.batch_modify(enc_batch)
        #print(encoder_inputs_.shape)
        max_sentence_length = encoder_inputs_.shape[0]
        batch_size = encoder_inputs_.shape[-1]
        decoder_targets_, _ = self.batch_modify(
            [(sequence) + [self.config.EOS] + [self.config.PAD] * 2 for sequence in enc_batch] #additional 3 spaces
        )
	feed = {
            self.word_ids: encoder_inputs_, 
            self.sequence_lengths: encoder_input_lengths_,
            self.decoder_targets: decoder_targets_,}
        
        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths
        
        feed[self.labels] = decoder_targets_
        if labels is not None:
            labels, _ = pad_sequences(labels, self.config.vocab_tags['O'])
            feed[self.labels] = labels
	
        if lr is not None:
            feed[self.lr] = self.config.lr

        if dropout is not None:
            feed[self.dropout] = dropout
            
        if (self.config.use_seq2seq):
            #feed[self.decoder_targets] = encoder_inputs_
            if(not self.config.train_seq2seq):#Seq2seq has been trained-> Use for absa task
                np_mask_matrix = np.ones((max_sentence_length, max_sentence_length))
                
                a = np.array(range(max_sentence_length))
                np_mask_matrix[np.arange(len(a)),a] = 0
                #NOTE: When for ABSA task we feed encoder, we feed as normal (the tf graph converts to time major format)
                feed[self.word_ids] = encoder_inputs_.swapaxes(0,1)
                
            else:
                np_mask_matrix = np.ones((1,1))
            
	    np_mask_matrix = np_mask_matrix.astype(bool)
                #shape=(len(words))
            feed[self.ones] = np.ones(shape=(batch_size), dtype="int32")
            feed[self.mask_matrix] =  np_mask_matrix
            feed[self.max_sentence_length] = max_sentence_length     
            
        
        return feed
                     
    def merge_feeds(self,df1,df2):
	'''This is hard coded for seq2seq merging of feed dicts. It returns the feed dict concatenated for the following fields: word_ids, sequence_lengths, decoder_targets, labels, max_sentence_length'''
	
	df1[self.word_ids] 		= np.append(df1[self.word_ids],df2[self.word_ids],axis=1)
	df1[self.sequence_lengths] 	= np.append(df1[self.sequence_lengths],df2[self.sequence_lengths],axis=0)
	df1[self.decoder_targets]  	= np.append(df1[self.decoder_targets],df2[self.decoder_targets], axis=1)
	df1[self.labels] 		= np.append(df1[self.labels], df2[self.labels],axis=1)
	df1[self.max_sentence_length]   = max(df1[self.max_sentence_length],df2[self.max_sentence_length])
	#print(df1[self.word_ids].shape)
	#print(df1[self.sequence_lengths].shape)
	#print(df1[self.decoder_targets].shape)
	#print(df1[self.labels].shape)
	#print(df1[self.max_sentence_length].shape)
	return df1	 
                             
    def compute_seq2seq_acc(self, words, predictions):
	#accuracy = 
	return 5
