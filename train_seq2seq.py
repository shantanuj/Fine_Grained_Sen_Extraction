from custom_model.data_utils import CoNLLDataset
from custom_model.ner_model import NERModel
from custom_model.config import Config


def main():
    # create instance of config
    config = Config()

    # build model
    model = NERModel(config)
    model.build()
    if(config.model_already_exists):
	print("Training existing model")
        model.restore_session(config.dir_model) # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = CoNLLDataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    
    # train model
    model.train_seq2seq(train, dev)
    #model.train(train, dev) #NOTE: SMALL QUICK FIX FOR PROBLEM IN RESTORING Lstm after training seq2seq and using for absa
if __name__ == "__main__":
    main()
