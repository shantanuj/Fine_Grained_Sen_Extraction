import os


from .general_utils import get_logger
from .data_utils import get_word_embeddings, load_vocab, load_word_vocab, \
        get_processing_word

class Config():
    def __init__(self, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
#        model_already_exists = True


	 # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
#	elif(model_already_exists == False):
#	    input("Are you sure you want to overwrite previous model?")
        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_word_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords     = len(self.vocab_words)
        self.nchars     = len(self.vocab_chars)
        self.ntags      = len(self.vocab_tags)

        self.pad_token = '<PAD>'
        self.eos_token = '<END>'
        self.PAD = self.vocab_words[self.pad_token]
        self.EOS = self.vocab_words[self.eos_token]
	#print(self.vocab_tags)
       # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words,
                self.vocab_chars, lowercase=True, chars=self.use_chars)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_word_embeddings(self.filename_trimmed)
                if self.use_pretrained else None)



    # embeddings
    dim_word = 200
    dim_char = 100

    #>>>>>> WORD VECTOR FILES<<<<<<<<<<<

    #filename_glove = "data/Embeddings/glove.6B/glove.6B.{}d.txt".format(dim_word)
    # trimmed embeddings (created from glove_filename with build_data.py)
    #filename_trimmed = "data/Embeddings/Pruned/np_Restw2vec_200d_trimmed.npz"#data/Embeddings/Pruned/np_glove_{}d_trimmed.npz".format(dim_word)
    use_pretrained = True

    # dataset
    #>>>>>>>>>>>>>>>>>>Training and testing files<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/vocab_to_id.pkl"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs  	     = 400
    #nepochs          = 100
    #NOTE:
    dropout          = 0.7
    dropout_seq2seq  = 1
    batch_size       = 25
    seq2seq_batch_size = 50
    lr_method        = "adagrad"
    #lr               = 0.2 
    #lr_decay	     = 0.99
    lr              = 0.1 #Seq2seq 0.2 starting seems best for seq2seq
    lr_decay        = 0.99  #0.9 or 0.6 for absa 
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 100

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 400 # lstm on word embeddings
    seq2seq_enc_hidden_size = 75#100 #50
    condense_dims = 100 #
    #seq2seq_enc_hidden_size = 200
    seq2seq_dec_hidden_size = seq2seq_enc_hidden_size*2
 		

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU
    use_seq2seq = True#True #False #Does model use seq2seq
    
    #Seq2seq stuff
    use_condense_layer = True and use_seq2seq
    use_only_seq2seq = False and use_seq2seq
    use_cosine_sim = False and use_seq2seq and not use_condense_layer 
    use_only_cosine_sim = False and use_seq2seq
    
    use_GRU = False and use_seq2seq
    use_only_h = False and use_seq2seq #True
    #use_only_seq2seq = True
   #NOTE
    seq2seq_trained = True#True
    #seq2seq_trained=  False and use_seq2seq#Has seq2seq been trained

    #complete_autoencode_including_test = False
    complete_autoencode_including_test = True and use_seq2seq #We only do this once testing data is available
    train_seq2seq = not(seq2seq_trained) #Use model to train seq2seq
    #assert (train_seq2seq and use_seq2seq) or not(train_seq2seq and use_seq2seq)
    
    def gen_model_extra_str(hidden_size_lstm,use_crf,use_chars,use_seq2seq,use_condense_layer,condense_dims=0):
        s = "bilstm{}".format(hidden_size_lstm)
        if(use_crf):
            s+='_crf'
        if(use_chars):
            s+='_chars'
        if(use_seq2seq):
            s+='_seq2seq'
	if(use_condense_layer):
	    s+='_condense{}'.format(condense_dims)
	
        return s
    #NOTE:>>>>>>>>>>> general config<<<<<<<<<<<<<<<<<<
    #domain = domain_train = "Laptop"
    domain = domain_train = "Rest"
    #domain_test = "Laptop"
    domain_test = "Rest"
    embedding_name = "w2v"
    filename_trimmed = "data/Embeddings/Pruned/np_Restw2vec_200d_trimmed.npz"#data/Embeddings/Pruned/np_glove_{}d_trimmed.npz".format(dim_word) 
    #testing_vecs = "data/Embeddings 
    use_CPU_only = True#False#True
    #NOTE
    model_already_exists = False#True#os.path.isdir(dir_output)

    
    extra = gen_model_extra_str(hidden_size_lstm,use_crf, use_chars,use_seq2seq,use_condense_layer,condense_dims)
    if(use_seq2seq):
    	extra += '_'+str(seq2seq_enc_hidden_size)
    filename_dev = filename_test = "data/{}test_data.txt".format(domain_test)#"data/Resttest_data.txt"
    #filename_dev = filename_test =
    filename_train = "data/{}train_data.txt".format(domain_train)#"data/Resttrain_data.txt" # test

    if(use_seq2seq):
        dir_output = "results/tr{}_te{}_{}_{}/".format(domain_train, domain_test, embedding_name, extra)
    else:
	dir_output = "results/tr{}_te{}_{}_{}/".format(domain_train, domain_train, embedding_name, extra)
 
    print("Model dir: {}".format(dir_output))	
    if(not model_already_exists and os.path.exists(dir_output)):
	x= int(input("Existing model found. Create new model or train existing model  (1/0)?"))
	if(not bool(x)):
	    model_already_exists = True
        if(x not in [1,0]):
	    x = int(input("1 to create or 0 to overwrite"))
	elif(x in [1]):
	    x= int(input("Are you sure to reset model(1)")) 
    dir_model  = dir_output + "model.weights/"

    path_log   = dir_output + "log.txt"



