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

	print(self.vocab_tags)
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
    # filename_dev = "data/coNLL/eng/eng.testa.iob"
    # filename_test = "data/coNLL/eng/eng.testb.iob"
    # filename_train = "data/coNLL/eng/eng.train.iob"

    #>>>>>>>>>>>>>>>>>>Training and testing files<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    max_iter = None # if not None, max number of examples in Dataset

    # vocab (created from dataset with build_data.py)
    filename_words = "data/vocab_to_id.pkl"
    filename_tags = "data/tags.txt"
    filename_chars = "data/chars.txt"

    # training
    train_embeddings = False
    nepochs          = 25
    dropout          = 0.5
    batch_size       = 25
    lr_method        = "adagrad"
    lr               = 0.02
    lr_decay         = 0.9
    clip             = -1 # if negative, no clipping
    nepoch_no_imprv  = 8

    # model hyperparameters
    hidden_size_char = 100 # lstm on chars
    hidden_size_lstm = 300 # lstm on word embeddings

    # NOTE: if both chars and crf, only 1.6x slower on GPU
    use_crf = True # if crf, training is 1.7x slower on CPU
    use_chars = False # if char embedding, training is 3.5x slower on CPU

    def gen_model_extra_str(hidden_size_lstm,use_crf,use_chars):
	s = "bilstm{}".format(hidden_size_lstm)
	if(use_crf):
	    s+='_crf'
	if(use_chars):
	    s+='_chars'

	return s
    #NOTE:>>>>>>>>>>> general config<<<<<<<<<<<<<<<<<<
    domain = domain_train = "Rest"
    domain_test = "Rest"
    embedding_name = "Geo_200d"
    filename_trimmed = "data/Embeddings/Pruned/differential_Rest_200d.npz"#data/Embeddings/Pruned/np_glove_{}d_trimmed.npz".format(dim_word)
 
    use_CPU_only = True#False#True
    #NOTE
    model_already_exists = True#os.path.isdir(dir_output)
    extra = gen_model_extra_str(hidden_size_lstm,use_crf, use_chars)
    filename_dev = filename_test = "data/{}test_data.txt".format(domain_test)#"data/Resttest_data.txt"
    #filename_dev = filename_test =
    filename_train = "data/{}train_data.txt".format(domain_train)#"data/Resttrain_data.txt" # test


    dir_output = "results/{}_{}_{}/".format(domain_train, embedding_name, extra)
    
    dir_model  = dir_output + "model.weights/"

    path_log   = dir_output + "log.txt"


