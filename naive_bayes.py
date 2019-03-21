from preprocessing import load_data,process_data

import numpy as np
import pandas as pd
train_model = process_data.NB_CLASSIFIER(load_data.CORPUS_TRAIN_HAM,load_data.CORPUS_TRAIN_SPAM)
VOCAB,PRIOR,CONDITIONAL_PROBABILITY = train_model.train()
TEST_HAM= load_data.CORPUS_TEST_HAM
TEST_SPAM = load_data.CORPUS_TEST_SPAM
correct_classify=0
incorrect_classify=0
for d in TEST_HAM:
    print(d)
    class_name = train_model.classify(VOCAB,PRIOR,CONDITIONAL_PROBABILITY,d)
    if(class_name == 1):
        correct_classify = correct_classify + 1
    else:
        incorrect_classify = incorrect_classify + 1
for d in TEST_SPAM:
    class_name = train_model.classify(VOCAB,PRIOR,CONDITIONAL_PROBABILITY,d)
    if(class_name == 0):
        correct_classify = correct_classify + 1
    else:
        incorrect_classify = incorrect_classify + 1
accuracy = (correct_classify/(correct_classify+incorrect_classify)) * 100
print("accuracy of classifier is {}%".format(accuracy))
