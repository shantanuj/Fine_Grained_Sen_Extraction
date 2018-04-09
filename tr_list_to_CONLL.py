# Script to convert pickled training list obtained from XML_to_tr_list.py into CONLL format txt file
import pickle
domain = "Laptop"
tr_data_path = "./Laptop_tr_list.pkl"
tr_split = 85
tr_output_path = "./{}_".format(tr_domain)

def tr_pkl_to_CONLL(domain, tr_data_path, tr_split = 85, tr_output_path = "./"):
    output_path_tr= tr_output_path+"{}_training_data.txt".format(domain)
    output_path_dev = tr_output_path+"{}_dev_data.txt".format(domain)
    tr_list = pickle.load(open(tr_data_path,'r'))
    training_last_index = int((tr_split/100.0)*len(tr_list))
    print(training_last_index)
    training_list = tr_list[:training_last_index]
    dev_list = tr_list[training_last_index:]
    
    with open(output_path_tr,'w') as f1:
        for words,tags in training_list:
            for i, word in enumerate(words):
                f1.write(word + " "+ tags[i]+"\n")
            f1.write("\n")
    print("Written output for training file to: {}".format(output_path_tr))
    with open(output_path_dev,'w') as f1:
        for words,tags in dev_list:
            for i, word in enumerate(words):
                f1.write(word + " "+tags[i]+"\n")
            f1.write("\n")
    print("Written output for dev file to: {}".format(output_path_dev))    
    
    
    
tr_pkl_to_CONLL(domain, tr_data_path)