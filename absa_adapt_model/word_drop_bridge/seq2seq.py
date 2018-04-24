#Dependencies and change directory
import os
import tensorflow as tf
from data_utils import get_word_embeddings
import pickle
import numpy as np
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple


domain_name = 'laptop'
domain_tr_data_path = '../data/Final_joint_data_absa//Domains/Laptop/Normal__normal_training_list.pickle'
embeddings_path = '../data/Embeddings/Pruned/np_glove_200d_trimmed.npz'
embeddings_name = 'glove200d'
vocab_path = '../data/vocab_to_id.pkl'



#Data load functions

def dataset_load(domain_tr_data_path, vocab_path):
    with open(domain_tr_data_path,'r') as p1:
        domain_tr_data = pickle.load(p1)
    with open(vocab_path,'r') as p1:
        vocab = pickle.load(p1)
        
    domain_tr_data = map(lambda x: x[0],domain_tr_data)
    idd_domain_tr_data = map(lambda x: [vocab[word] for word in x], domain_tr_data)
    return idd_domain_tr_data, vocab


def get_word_embeddings(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)
        
#Data stream for model 
def gen_batch(idd_data, batch_size):
    np.random.shuffle(idd_data)
    rem = len(idd_data)%batch_size
    num_batches = (len(idd_data)/batch_size) 
    if rem>0:
        num_batches = num_batches + 1
    
    
    for i in range(num_batches):
        if(i == num_batches -1 and (not rem==0)):
            yield(idd_data[i*batch_size:])
        else:
            yield(idd_data[i*batch_size:(i+1)*batch_size])
            
def batch_modify(inputs, max_sequence_length=None):
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
    
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32) # == PAD
    
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths
        
def next_feed(batch, use_decoder = True):
    encoder_inputs_, encoder_input_lengths_ = batch_modify(batch)
    if(use_decoder):
        decoder_targets_, _ = batch_modify(
            [(sequence) + [EOS] + [PAD] * 3 for sequence in batch] #additional 3 spaces
        )
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
        }
    else:
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
        }
            

def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished) # -> boolean scalar
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None

    return (elements_finished, 
            input,
            state,
            output,
            loop_state)

def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)


def predict_batch():
    None



    
#The model 


encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')

encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    
with tf.variables_scope("embeddings"):
    embeddings = tf.Variable(word_embeddings_np, name="word_embeds",dtype=tf.float32, trainable=False)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    
with tf.variables_scope("encoder"):
    encoder_cell = LSTMCell(encoder_hidden_units)
    ((encoder_fw_outputs,
      encoder_bw_outputs),
     (encoder_fw_final_state,
      encoder_bw_final_state)) = (
        tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
    )
    
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs),2)

    encoder_final_state_c = tf.concat(
        (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

    encoder_final_state_h = tf.concat(
        (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

    encoder_final_state = LSTMStateTuple(
        c=encoder_final_state_c,
        h=encoder_final_state_h
    ) #this is useful later
    
    encoder_concat_everything = tf.concat([encoder_final_state_c,encoder_final_state_h], 1)
    
    
with tf.variables_scope("decoder"):
    decoder_cell = LSTMCell(decoder_hidden_units)
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
    decoder_lengths = encoder_inputs_length + 4 #3 additional 
    
with tf.variables_scope("projection"):
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
    
    
 


'''Start computation: Load data first'''
pad_token = '<START>' #This is due to a mistake on my part. Didn't store PAD in vocab, stored unkown-> might lead to ambiguity. So assume start == pad for now. Will edit vocab later
eos_token = '<END>'
PAD = vocab[pad_token]
EOS = vocab[eos_token]

vocab_size = len(vocab)
input_embedding_size = 200
encoder_hidden_units = 100
decoder_hidden_units = 100*2



eos_time_slice = EOS*tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = PAD*tf.ones([batch_size], dtype=tf.int32, name='PAD')

eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs = decoder_outputs_ta.stack()

decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
decoder_prediction = tf.argmax(decoder_logits, 2)


stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)

    
    










idd_data = dataset_load(domain_tr_data_path,vocab_path)

loss_track = []


batch_size = 25
num_epochs = 10
embed_type = "Glove"
model_path = '../results/seq2seq/{}_seq2seqmodel_embeds{}_{}d_{}hiddenunits'.format(domain_name,embed_type,input_embedding_size,encoder_hidden_units)

batch_size = 30
lr = None
num_epochs = 1


if (lr is None):
    train_op = tf.train.AdamOptimizer().minimize(loss)
else:
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


if not os.path.exists(model_path):
    os.makedirs(model_path)
    
    
saver = tf.train.Saver()
init = tf.global_variables_initializer()
config = tf.ConfigProto(device_count={'GPU': 0})

def train_graph():
    with tf.Session(config=config) as sess:
        sess.run(init)
        print("Initialized session")
        for epoch in range(num_epochs):
            print("At epoch: {}".format(epoch))
            iters = 0 
            epoch_loss = 0.
            batch_generator = gen_batch(idd_data, batch_size)
            for batch in batch_generator:
                
                fd = next_feed(batch)
                _, l = sess.run([train_op, loss], fd)
                if(iters%33==0):
                    print("At batch: {}".format(iters))
                    print("Batch loss: {}".format(l))
                loss_track.append(l)
                iters+=1
                epoch_loss+=l
                encoder_useful_state = sess.run(encoder_concat_everything, fd)
            print("Epoch training loss: {}".format(epoch_loss/iters))
        saver.save(sess,model_path)
        print("Saved model at: {}".format(model_path))
        #encoder_useful_state = sess.run(encoder_concat_everything)
    return encoder_useful_state


encoder_out = train_graph()

