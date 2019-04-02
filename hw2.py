# CISC684 2019Spring
# HW2 Group Assignment
# Anantvir Singh
# Anuj Gandhi
# Hang Chen
# Siqi Wang

import os
import sys
import re
import collections
from copy import deepcopy
import random
import math

# trainig_set_dir = sys.argv[1]
# test_set_dir = sys.argv[2]
# stop_word_dir = sys.argv[3]

trainig_set_dir = '/Users/chenhang91/TEMP/684group2/hw 2 datasets/dataset 1/train/'
test_set_dir = '/Users/chenhang91/TEMP/684group2/hw 2 datasets/dataset 1/test/'
stop_word_path = '/Users/chenhang91/TEMP/684group2/hw 2 datasets/stop_words.txt' # found from some github that contains common stop words to skip

def process_documents_and_extract_words(data_set_dir):

    print("Processing documents, please wait...")
    
    whole_data_set_doc_by_doc = {} # used to extract the number of the word appearances for each document
    divided_training_set = {} # 70% of the original training set used in Preceptron and Logistic Regression
    divided_validation_set = {} # 30% of the original training set as the validation set used in Preceptron and Logistic Regression
    data_set_by_class_label = {} # used to extract the number of the word appearances for each class
    vocabulary_set = set() # used to extract the whole vacabulary from the document spaces
    stop_words = set()
    class_labels = []

    # read in stop_words first
    if os.path.isfile(stop_word_path):
        with open(stop_word_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
            stop_words = set(re.findall(r'\w+', txt_file)) # https://www.guru99.com/python-regular-expressions-complete-tutorial.html#2

    for class_entry in os.listdir(data_set_dir):
        if not class_entry.startswith('.') and os.path.isdir(os.path.join(data_set_dir, class_entry)): #https://stackoverflow.com/questions/3761473/python-not-recognising-directories-os-path-isdir #https://stackoverflow.com/questions/15235823/how-to-ignore-hidden-files-in-python-functions
            print(class_entry)
            if not class_entry.startswith('.'):
                print(class_entry)
            class_labels.append(class_entry)
            whole_data_set_doc_by_doc[class_entry] = {}
            data_set_by_class_label[class_entry] = {}
            divided_training_set[class_entry] = {}
            divided_validation_set[class_entry] = {}
            num_of_original_training_instances = len([doc for doc in os.listdir(data_set_dir + class_entry) if not doc.startswith('.') and os.path.isfile(os.path.join(data_set_dir, class_entry, doc))]) # count docs in the folder to prepare to split the data # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python
            training_instance_devide_index = int(num_of_original_training_instances * 0.7) # we select the first 70% of the original training data as the training set. We didn't randomly choose the 70% because we have to run this program various of times to test for hyper-parameter so we have to have the same training_set for each time
            training_set_index_iter = 0
            for doc_entry in os.listdir(data_set_dir + class_entry):
                if not doc_entry.startswith('.') and os.path.isfile(os.path.join(data_set_dir, class_entry, doc_entry)):
                    with open(os.path.join(data_set_dir, class_entry, doc_entry), 'r', encoding='utf-8', errors='ignore') as doc_file:
                        doc_content = doc_file.read()
                        training_set_index_iter += 1
                        words_dict_for_this_doc = collections.Counter(re.findall(r'\w+', doc_content)) # a (Counter) dictionary that counts the words # https://stackoverflow.com/questions/11011756/is-there-any-pythonic-way-to-combine-two-dicts-adding-values-for-keys-that-appe
                        for key in words_dict_for_this_doc: # remove stop words
                            if key in stop_words:
                                del words_dict_for_this_doc[key]
                        # add words to whole_data_set_doc_by_doc
                        whole_data_set_doc_by_doc[class_entry][doc_entry] = words_dict_for_this_doc
                        # add words to training or validation set
                        if training_set_index_iter < training_instance_devide_index:
                            divided_training_set[class_entry][doc_entry] = words_dict_for_this_doc 
                        else:
                            divided_validation_set[class_entry][doc_entry] = words_dict_for_this_doc
                        # add words to data_set_by_class_label
                        data_set_by_class_label[class_entry] = collections.Counter(data_set_by_class_label[class_entry]) + words_dict_for_this_doc
                        # add words to the whole vocabulary
                        vocabulary_set.update(set(words_dict_for_this_doc.keys()))     
                else:
                    continue
        else:
            sys.exit("Please follow the path entry instruction. Program aborts.") #https://stackoverflow.com/questions/179369/how-do-i-abort-the-execution-of-a-python-script
    print("Documents processed!\n")
    return whole_data_set_doc_by_doc, divided_training_set, divided_validation_set, data_set_by_class_label, vocabulary_set, class_labels

whole_data_set_doc_by_doc, divided_training_set, divided_validation_set, data_set_by_class_label, vocabulary_set, class_labels = process_documents_and_extract_words(trainig_set_dir)
print()

''' used whole_data_set_doc_by_doc to count for documents under each class;
    used vocabulary_set to iterate each word;
    used data_set_by_class_label to calculate the conditional probability for each word;
    used algorithm in 13bayes page260
    '''
def navie_bayes():
    print("Learning by Naive Bayes...")
    prior = {} # count of the documents under each class
    conditional_prob_for_each_word = {}
    num_of_unique_words_in_all_document_space = len(vocabulary_set)
    for class_label in whole_data_set_doc_by_doc:
        prior[class_label] = len(whole_data_set_doc_by_doc[class_label])
        entire_docs_count += prior[class_label]
    for class_count in prior:
        prior[class_count] /= entire_docs_count
    for class_label in data_set_by_class_label:
        total_num_of_words_in_this_class = sum(data_set_by_class_label[class_label].values()) # https://stackoverflow.com/questions/4880960/how-to-sum-all-the-values-in-a-dictionary
        class_label_counts_with_smoothing = total_num_of_words_in_this_class + num_of_unique_words_in_all_document_space # used as the denominator when calculating the conditional probability of a word
        for word in vocabulary_set:
            if word not in data_set_by_class_label[class_label]:
                conditional_prob_for_each_word[word] = (0 + 1) / class_label_counts_with_smoothing
            else:
                conditional_prob_for_each_word[word] = (data_set_by_class_label[class_label][word] + 1) / class_label_counts_with_smoothing
    print("Naive Bayes learning finished.\n")
    return prior, conditional_prob_for_each_word

''' used whole_data_set_doc_by_doc to iterate over words in each document(since perceptron is an
    incremental algorithm);
    used algorithm in Lec7 page6'''
def perceptron(learning_rate, training_iterations):
    print("Learning by Perceptron Rule...")
    weight_vector = {'weight_bias': 0.1}
    for i in range(training_iterations):
        for class_label in whole_data_set_doc_by_doc:
            for document in whole_data_set_doc_by_doc[class_label]:
                prediction = ""
                perceptron_weighted_sum_of_this_doc = weight_vector['weight_bias']
                for word in whole_data_set_doc_by_doc[class_label][document]:
                    if word not in weight_vector:
                        weight_vector[word] = random.uniform(0.01, 0.1) # initialize weights as some small values # https://stackoverflow.com/questions/6088077/how-to-get-a-random-number-between-a-float-range
                    perceptron_weighted_sum_of_this_doc += weight_vector[word] * whole_data_set_doc_by_doc[class_label][document][word]
                # sign function
                prediction = class_labels[0] if perceptron_weighted_sum_of_this_doc > 0 else class_labels[1] # https://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator conditional operator in python
                if prediction != class_label:
                    # update weight, we treat class_labels[0] as 1 and class_labels[1] as -1
                    target_val_minus_prediction = (1 - (-1)) if class_label == class_labels[0] else (-1 - 1)
                    for word in whole_data_set_doc_by_doc[class_label][document]:
                        weight_vector[word] += learning_rate * target_val_minus_prediction * whole_data_set_doc_by_doc[class_label][document][word]
                else:
                    continue
    return weight_vector

'''used algorithm in Lec6 page26 and page30'''
def logistic_regression(learning_rate, regularization_lambda, training_iterations):
    print("Learning by Logistic Regression...")
    weight_vector = {'weight_bias': 0.1}
    for i in range(training_iterations):
        # initialize weights for all words
        for word in vocabulary_set:
            weight_vector[word] = random.uniform(0.01, 0.1)
        # calculate the weighted sum
        weighted_sum_of_this_doc = weight_vector['weight_bias']
        for class_label in whole_data_set_doc_by_doc:
            for document in whole_data_set_doc_by_doc[class_label]:
                for word in whole_data_set_doc_by_doc[class_label][document]:
                    weighted_sum_of_this_doc += weight_vector[word] * whole_data_set_doc_by_doc[class_label][document][word]
        # update the weight
        yi = 1 if class_label == class_labels[0] else -1
        exp_weighted_sum_of_this_doc = math.exp(weighted_sum_of_this_doc)
        weight_vector[word] += learning_rate * whole_data_set_doc_by_doc[class_label][document][word] * (yi - exp_weighted_sum_of_this_doc/(1 + exp_weighted_sum_of_this_doc)) - learning_rate * regularization_lambda * weight_vector[word]
    return weight_vector





