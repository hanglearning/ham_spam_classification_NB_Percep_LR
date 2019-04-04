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
import operator

trainig_set_dir = os.path.join(os.getcwd(), sys.argv[1])
test_set_dir = os.path.join(os.getcwd(), sys.argv[2])
stop_word_path = os.path.join(os.getcwd(), sys.argv[3]) # found from some github that contains common stop words to skip

def process_data_and_extract_words(training_set_dir, test_set_dir):

    print("Processing documents, please wait...")
    
    whole_training_set_doc_by_doc = {} # used to extract the number of the word appearances for each document in the origianl training set
    divided_training_set = {} # 70% of the original training set used in Preceptron and Logistic Regression to find the hyperparams
    divided_validation_set = {} # 30% of the original training set as the validation set used in Preceptron and Logistic Regression
    whole_training_set_word_count_by_class_label = {} # used to extract the number of the word appearances for each class. useful in naive bayes
    vocabulary_set = set() # used to extract the whole vacabulary from the document spaces
    vocabulary_set_divided_training_set = set() # corresponding to the vocabulary in the first 70% training set
    stop_words = set()
    class_labels = [] # used to record the text labels
    whole_test_set_doc_by_doc = {} # used while testing on the whole test set

    # read in stop_words first
    if os.path.isfile(stop_word_path):
        with open(stop_word_path, 'r', encoding='utf-8', errors='ignore') as txt_file:
            stop_words = set(re.findall(r'\w+', txt_file.read().lower()))

    # processing the whole original training set data
    for class_entry in os.listdir(training_set_dir):
        if not class_entry.startswith('.') and os.path.isdir(os.path.join(training_set_dir, class_entry)):

            class_labels.append(class_entry)
            whole_training_set_doc_by_doc[class_entry] = {}
            whole_training_set_word_count_by_class_label[class_entry] = {}
            divided_training_set[class_entry] = {}
            divided_validation_set[class_entry] = {}

            # count total docs in the training data and get a division index used to control the selection of the first 70% of the original training data as the training set
            training_instance_divide_index = int(len([doc for doc in os.listdir(os.path.join(training_set_dir, class_entry)) if not doc.startswith('.') and os.path.isfile(os.path.join(training_set_dir, class_entry, doc))]) * 0.7) 

            training_set_index_iter = 0
            for doc_entry in os.listdir(os.path.join(training_set_dir, class_entry)):
                if not doc_entry.startswith('.') and os.path.isfile(os.path.join(training_set_dir, class_entry, doc_entry)):
                    with open(os.path.join(training_set_dir, class_entry, doc_entry), 'r', encoding='utf-8', errors='ignore') as doc_file:
                        training_set_index_iter += 1
                        words_count_for_this_doc = collections.Counter(re.findall(r'\w+', doc_file.read().lower())) # a (Counter) dictionary that counts the words 
                        words_count_for_this_doc_keys = list(words_count_for_this_doc.keys())
                        for key in words_count_for_this_doc_keys:
                            # remove the stop words
                            if key in stop_words:
                                del words_count_for_this_doc[key]
                        # add words to whole_training_set_doc_by_doc
                        whole_training_set_doc_by_doc[class_entry][doc_entry] = words_count_for_this_doc
                        # add words to 70% training or 30% validation set
                        if training_set_index_iter <= training_instance_divide_index:
                            divided_training_set[class_entry][doc_entry] = words_count_for_this_doc
                            # add words to the divided training_set vocabulary set
                            vocabulary_set_divided_training_set.update(set(words_count_for_this_doc.keys()))
                        else:
                            divided_validation_set[class_entry][doc_entry] = words_count_for_this_doc
                        # add words to whole_training_set_word_count_by_class_label
                        whole_training_set_word_count_by_class_label[class_entry] = collections.Counter(whole_training_set_word_count_by_class_label[class_entry]) + words_count_for_this_doc
                        # add words to the whole vocabulary of the original training data
                        vocabulary_set.update(set(words_count_for_this_doc.keys()))     
                else:
                    continue
        else:
            continue

    # processing the whole test set data
    for class_entry in os.listdir(test_set_dir):
        if not class_entry.startswith('.') and os.path.isdir(os.path.join(test_set_dir, class_entry)):
            whole_test_set_doc_by_doc[class_entry] = {}
            for doc_entry in os.listdir(os.path.join(test_set_dir, class_entry)):
                if not doc_entry.startswith('.') and os.path.isfile(os.path.join(test_set_dir, class_entry, doc_entry)):
                    with open(os.path.join(test_set_dir, class_entry, doc_entry), 'r', encoding='utf-8', errors='ignore') as doc_file:
                        words_count_for_this_doc = collections.Counter(re.findall(r'\w+', doc_file.read().lower()))
                        words_count_for_this_doc_keys = list(words_count_for_this_doc.keys())
                        for key in words_count_for_this_doc_keys:
                            if key in stop_words:
                                del words_count_for_this_doc[key]
                        whole_test_set_doc_by_doc[class_entry][doc_entry] = words_count_for_this_doc
                else:
                    continue
        else:
            continue

    print("Documents processed!\n")
    return whole_training_set_doc_by_doc, divided_training_set, divided_validation_set, whole_training_set_word_count_by_class_label, vocabulary_set, class_labels, stop_words, vocabulary_set_divided_training_set, whole_test_set_doc_by_doc

try:
    whole_training_set_doc_by_doc, divided_training_set, divided_validation_set, whole_training_set_word_count_by_class_label, vocabulary_set, class_labels, stop_words, vocabulary_set_divided_training_set, whole_test_set_doc_by_doc = process_data_and_extract_words(trainig_set_dir, test_set_dir)
except:
    sys.exit("Please follow the commend line path entry instruction. Program aborts.")

''' used whole_training_set_doc_by_doc to count for documents under each class;
    used vocabulary_set to iterate over each word;
    used whole_training_set_word_count_by_class_label to calculate the conditional probability for each word;
    Originally used algorithm in 13bayes page260, but the smoothing method is not consistent with the one used in Lec5 page26. We finally adapted the machanism from Lec5 since that makes more sense to us. '''
def train_navie_bayes():
    print("Learning by Naive Bayes...")
    naive_bayes_class_prior = {} # count of the documents under each class
    num_of_words_occurances = {}
    num_of_words_occurances['num_of_unique_words_in_all_document_space'] = len(vocabulary_set)
    # calculate naive_bayes_class_prior
    entire_docs_count = 0
    for class_label in whole_training_set_doc_by_doc:
        naive_bayes_class_prior[class_label] = len(whole_training_set_doc_by_doc[class_label])
        entire_docs_count += naive_bayes_class_prior[class_label]
    for class_count in naive_bayes_class_prior:
        naive_bayes_class_prior[class_count] /= entire_docs_count
    # sum over all the number of word occurances in one particular class 
    for class_label in whole_training_set_word_count_by_class_label:
        num_of_words_occurances[class_label] = sum(whole_training_set_word_count_by_class_label[class_label].values())
    print("Naive Bayes learning accomplished!\n")
    return naive_bayes_class_prior, num_of_words_occurances

''' used algorithm in Lec7 page6;
    training_data_set should be either the 70% divided training set or the whole training data set. '''
def train_perceptron(learning_rate, training_iterations, training_data_set):
    print("Learning by Perceptron Rule...")
    weight_vector = {'bias_term': 0.1}
    for i in range(training_iterations):
        for class_label in training_data_set:
            for document in training_data_set[class_label]:
                prediction = ""
                weighted_sum_of_this_doc = weight_vector['bias_term'] * 1
                for word in training_data_set[class_label][document]:
                    if word not in weight_vector:
                        # initialize weights as some small values
                        weight_vector[word] = random.uniform(0.001, 0.01)
                    weighted_sum_of_this_doc += weight_vector[word] * training_data_set[class_label][document][word]
                # sign function
                prediction = class_labels[0] if weighted_sum_of_this_doc > 0 else class_labels[1]
                if prediction != class_label:
                    # update weight, we treat class_labels[0] as 1 and class_labels[1] as -1
                    target_val_minus_prediction = (1 - (-1)) if class_label == class_labels[0] else (-1 - 1)
                    for word in training_data_set[class_label][document]:
                        weight_vector[word] += learning_rate * target_val_minus_prediction * training_data_set[class_label][document][word]
                    # update weight for the bias term
                    weight_vector['bias_term'] += learning_rate * target_val_minus_prediction * 1
                else:
                    continue
    print("Learning by Perceptron Rule accomplished!\n")
    return weight_vector

'''used algorithm in Lec6 page26 and page30'''
def train_logistic_regression(learning_rate, regularization_lambda, training_iterations, training_data_set, learn_on_divided_training_set=None):
    print("Learning by Logistic Regression...")
    vocabulary_set_used = vocabulary_set_divided_training_set if learn_on_divided_training_set == True else vocabulary_set
    weight_vector = {'bias_term': 0.1}
    # initialize weights for all words
    for word in vocabulary_set_used:
        weight_vector[word] = random.uniform(0.001, 0.01)
    weight_vector_to_update = deepcopy(weight_vector) # because LR is a batch algorithm, we should only update the weights of the features after a full run of iteration
    for i in range(training_iterations):
        for word in vocabulary_set_used:
            for class_label in training_data_set:
                for document in training_data_set[class_label]:
                    if word in training_data_set[class_label][document]:
                        weighted_sum_of_this_doc = weight_vector['bias_term'] * 1
                        for word_in_this_doc in training_data_set[class_label][document]:
                            # calculate the weighted sum
                            weighted_sum_of_this_doc += weight_vector[word_in_this_doc] * training_data_set[class_label][document][word_in_this_doc]
                        # prepare the vars tp update the weight
                        yi = 1 if class_label == class_labels[0] else 0
                        exp_weighted_sum_of_this_doc = math.exp(weighted_sum_of_this_doc)
                        # update the weights for all the features
                        weight_vector_to_update[word] += learning_rate * training_data_set[class_label][document][word] * (yi - exp_weighted_sum_of_this_doc/(1 + exp_weighted_sum_of_this_doc)) - learning_rate * regularization_lambda * weight_vector[word]
                    else:
                        continue
        # update the weight for the bias term
        for class_label in training_data_set:
            for document in training_data_set[class_label]:
                for word_in_this_doc in training_data_set[class_label][document]:
                        weighted_sum_of_this_doc += weight_vector[word_in_this_doc] * training_data_set[class_label][document][word_in_this_doc]
                yi = 1 if class_label == class_labels[0] else 0
                exp_weighted_sum_of_this_doc = math.exp(weighted_sum_of_this_doc)
                weight_vector_to_update['bias_term'] += learning_rate * (yi - exp_weighted_sum_of_this_doc/(1 + exp_weighted_sum_of_this_doc)) #no regularization for the bias term
        weight_vector = deepcopy(weight_vector_to_update)
        for word in weight_vector:
            weight_vector[word] /= 10000 # to avoid exp() explode
    return weight_vector

def classify_and_test_for_accuracy(classifier, test_on_validation=None, perceptron_weight_vector=None, logistic_regression_weight_vector=None):
    print("Testing by {}...".format(classifier))
    correctly_cassified = 0
    test_set_used = divided_validation_set if test_on_validation == True else whole_test_set_doc_by_doc
    vocabulary_set_used = vocabulary_set_divided_training_set if test_on_validation == True else vocabulary_set
    documents_count = 0
    for class_entry in test_set_used:
        documents_count += len(test_set_used[class_entry])
        for doc_entry in test_set_used[class_entry]:
            prediction = ""
            if classifier == "Naive Bayes":
                class_score_for_this_doc = {}
                for class_label in class_labels:
                    class_score_for_this_doc[class_label] = math.log(naive_bayes_class_prior[class_label])
                    for word in test_set_used[class_entry][doc_entry]:
                        if word not in whole_training_set_word_count_by_class_label[class_label]:
                            whole_training_set_word_count_by_class_label[class_label][word] = 0
                        # apply smoothing
                        class_score_for_this_doc[class_label] += test_set_used[class_entry][doc_entry][word] * math.log((whole_training_set_word_count_by_class_label[class_label][word] + 1)/ num_of_words_occurances[class_label] + num_of_words_occurances['num_of_unique_words_in_all_document_space'])
                prediction = max(class_score_for_this_doc.items(), key=operator.itemgetter(1))[0] # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
            elif classifier == "Perceptron":
                perceptron_weighted_sum_of_this_doc = perceptron_weight_vector['bias_term'] * 1
                for word in test_set_used[class_entry][doc_entry]:
                    if word in vocabulary_set_used:
                        perceptron_weighted_sum_of_this_doc += perceptron_weight_vector[word] * test_set_used[class_entry][doc_entry][word]
                    else:
                        # we ignore the unseen words from the training sets
                        continue
                # sign function
                prediction = class_labels[0] if perceptron_weighted_sum_of_this_doc > 0 else class_labels[1]
            elif classifier == "Logistic Regression":
                logistic_regression_weighted_sum_of_this_doc = logistic_regression_weight_vector['bias_term'] * 1
                for word in test_set_used[class_entry][doc_entry]:
                    if word in vocabulary_set_used:
                        logistic_regression_weighted_sum_of_this_doc += logistic_regression_weight_vector[word] * test_set_used[class_entry][doc_entry][word]
                        exp_weighted_sum_of_this_doc = math.exp(logistic_regression_weighted_sum_of_this_doc)
                        # sigmoid function
                        prediction = class_labels[1] if 1/(1 + exp_weighted_sum_of_this_doc) > exp_weighted_sum_of_this_doc/(1 + exp_weighted_sum_of_this_doc) else class_labels[0] # corresponding to yi = 1 if class_label == class_labels[0] else 0 in train_logistic_regression
                    else:
                        continue
            else:
                sys.exit("Please input a valid classifier to test the accuracy. Program aborts.")
            correctly_cassified += 1 if prediction == class_entry else 0
    
    test_set_name = "validation" if test_on_validation == True else "whole test"
    print("Accuracy on the {} set by {} is {}/{} ({:.0%})".format(test_set_name, classifier, correctly_cassified, documents_count, correctly_cassified/documents_count)) 
    return correctly_cassified

# Test Naive Bayes
# naive_bayes_class_prior, num_of_words_occurances = train_navie_bayes()
# classify_and_test_for_accuracy("Naive Bayes")

# Choosing the best lambda for logistic regression
print("\nChoosing the best regularization constant lambda for Logistic Regression...")
print("We will use 100 training iterations and the learning rate 0.003, and test for lambda in [0, 10]")
learning_rate = 0.003
training_iterations = 100
regularization_lambda_with_accuracy = {}
for regularization_lambda in range(11):
    print("Using lambda {} on the 70% training set...".format(regularization_lambda))
    logistic_regression_weight_vector = train_logistic_regression(learning_rate, regularization_lambda, training_iterations, divided_training_set, True)
    print("Testing lambda {} on the 30% validation set...".format(regularization_lambda))
    regularization_lambda_with_accuracy[regularization_lambda] = classify_and_test_for_accuracy(classifier="Logistic Regression", test_on_validation=True, logistic_regression_weight_vector=logistic_regression_weight_vector)
    print()
best_lambda = max(regularization_lambda_with_accuracy.items(), key=operator.itemgetter(1))[0]
print("After testing on the validation set, the best to choose is {}".format(best_lambda))
print("Using lambda {} to train on the training set with the same iterations and learning rate.".format(best_lambda))
logistic_regression_weight_vector = train_logistic_regression(learning_rate, best_lambda, training_iterations, whole_training_set_doc_by_doc)
print("Using lambda {} to test on the test set".format(best_lambda))
classify_and_test_for_accuracy(classifier="Logistic Regression", logistic_regression_weight_vector=logistic_regression_weight_vector)

# # Choosing the best hyperparameter learning_rate and training_iterations for perceptron
# print("\nChoosing the best learning_rate and training_iterations for perceptron...")
# print("We will use training iterations in (10, 20, 50) and the learning rate (0.003, 0.01, 0.03), report the accuracy, and use the best params to test on the test data")
# iterations_and_rate_with_accuracy = {}
# for training_iterations in (10, 20, 50):
#     for learning_rate in (0.003, 0.01, 0.03):
#         print("Using training iterations {} and learning_rate {} on the 70% training set...".format(training_iterations, learning_rate))
#         perceptron_weight_vector = train_perceptron(learning_rate, training_iterations, divided_validation_set)
#         print("Using training iterations {} and learning_rate {} on the 30% validation set...".format(training_iterations, learning_rate))
#         iterations_and_rate_with_accuracy[training_iterations] = {}
#         iterations_and_rate_with_accuracy[training_iterations][learning_rate] = classify_and_test_for_accuracy("Perceptron", test_on_validation=True, divided_validation_set=divided_validation_set, perceptron_weight_vector=perceptron_weight_vector)
#         print()
# print("After testing on the validation set, the best to choose is any. So we will choose training_iterations 50 and learning_rate 0.003 to test on the whole test set.")
# perceptron_weight_vector = train_perceptron(learning_rate=0.003, training_iterations=50, training_data_set=whole_training_set_doc_by_doc)
# classify_and_test_for_accuracy(classifier="Perceptron", perceptron_weight_vector=perceptron_weight_vector)








