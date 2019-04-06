This repo contains the code for HW2 of our Machine Learning class. In this assignment we are required to write three classifiers - Naive Bayes, Logistic Regreesion and Perceptron to perform Spam/Ham email text classification task.

We are also required to test the best hyperparameters possible by testing on the validation data for our classifiers and use those params to re-learn on the whole training set and test on the whole test set.

A detailed accuracy report is also included within this repo.

How to run our code:

Please run from the command line -

$ python hw2.py <training_set_path> <test_set_paht> <stop_words.txt_path>

For example:

$ python hw2.py hw\ 2\ datasets/dataset\ 1/train/ hw\ 2\ datasets/dataset\ 1/test/ stop_words.txt

The stop_words.txt was found from 
https://github.com/henrydinh/Logistic-Regression-Text-Classification
Thanks to henrydinh!
