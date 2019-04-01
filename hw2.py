# CISC684 2019Spring
# HW2 Group Assignment
# Anantvir Singh
# Anuj Gandhi
# Hang Chen
# Siqi Wang

import os
import re
import collections
from copy import deepcopy

# trainig_set_dir = sys.argv[1]
# test_set_dir = sys.argv[2]
# stop_word_dir = sys.argv[3]

trainig_set_dir = '/Users/chenhang91/TEMP/684group2/hw 2 datasets/dataset 1/train/'
test_set_dir = '/Users/chenhang91/TEMP/684group2/hw 2 datasets/dataset 1/test/'
stop_word_path = '/Users/chenhang91/TEMP/684group2/hw 2 datasets/stop_words.txt' # found from some github that contains common stop words to skip

def process_documents_and_extract_words(data_set_dir):

    print("Processing documents, please wait...")
    
    data_set_doc_by_doc = {} # used to extract the number of the word appearances for each document
    data_set_by_class_label = {} # used to extract the number of the word appearances for each class
    vocabulary_set = set() # used to extract the whole vacabulary from the document spaces
    stop_words = set()

    # read in stop_words first
    if os.path.isfile(stop_word_path):
        with open(stop_word_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
            stop_words = set(re.findall(r'\w+', txt_file)) # https://www.guru99.com/python-regular-expressions-complete-tutorial.html#2

    for class_entry in os.listdir(data_set_dir):
        if os.path.isdir(os.path.join(data_set_dir, class_entry)): #https://stackoverflow.com/questions/3761473/python-not-recognising-directories-os-path-isdir
            data_set_doc_by_doc[class_entry] = {}
            data_set_by_class_label[class_entry] = {}
            for doc_entry in os.listdir(data_set_dir + class_entry):
                if os.path.isfile(os.path.join(data_set_dir, class_entry, doc_entry)):
                    with open(os.path.join(data_set_dir, class_entry, doc_entry), 'r', encoding='utf-8', errors='ignore') as doc_file:
                        doc_content = doc_file.read()
                        words_dict_for_this_doc = collections.Counter(re.findall(r'\w+', doc_content)) # a (Counter) dictionary that counts the words # https://stackoverflow.com/questions/11011756/is-there-any-pythonic-way-to-combine-two-dicts-adding-values-for-keys-that-appe
                        for key in words_dict_for_this_doc: # remove stop words
                            if key in stop_words:
                                del words_dict_for_this_doc[key]
                        # add words to data_set_doc_by_doc
                        data_set_doc_by_doc[class_entry][doc_entry] = words_dict_for_this_doc
                        # add words to data_set_by_class_label
                        data_set_by_class_label[class_entry] = collections.Counter(data_set_by_class_label[class_entry]) + words_dict_for_this_doc
                        # add words to the whole vocabulary
                        vocabulary_set.update(set(words_dict_for_this_doc.keys()))     
                else:
                    continue
        else:
            print("Please follow the path entry instruction. Program aborts.")
            return
    print("Documents processed!\n")
    return data_set_doc_by_doc, data_set_by_class_label, vocabulary_set

data_set_doc_by_doc, data_set_by_class_label, vocabulary_set = process_documents_and_extract_words(trainig_set_dir)

''' used data_set_doc_by_doc to count for documents under each class;
    used vocabulary_set to iterate each word;
    used data_set_by_class_label to calculate the conditional probability for each word
    '''
def navie_bayes():
    print("Learning by Naive Bayes...")
    prior = {} # count of the documents under each class
    conditional_prob_for_each_word = {}
    entire_docs_count = 0
    for class_label in data_set_doc_by_doc:
        prior[class_label] = len(data_set_doc_by_doc[class_label])
        entire_docs_count += prior[class_label]
    for class_count in prior:
        prior[class_count] /= entire_docs_count
    for word in vocabulary_set:
        

