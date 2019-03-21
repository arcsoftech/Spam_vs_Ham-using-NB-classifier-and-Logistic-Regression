import glob
import os
file_list_test_ham  =  glob.glob(os.path.join(os.getcwd(), "hw2_test","test","ham", "*.txt"))
file_list_test_spam =  glob.glob(os.path.join(os.getcwd(), "hw2_test","test","spam", "*.txt"))
file_list_train_ham  = glob.glob(os.path.join(os.getcwd(), "hw2_train","train","ham", "*.txt"))
file_list_train_spam = glob.glob(os.path.join(os.getcwd(), "hw2_train","train","spam", "*.txt"))

CORPUS_TEST_HAM = []
CORPUS_TEST_SPAM = []
CORPUS_TRAIN_HAM = []
CORPUS_TRAIN_SPAM = []


for file_path in file_list_test_ham:
    with open(file_path,encoding='latin1') as f_input:
        CORPUS_TEST_HAM.append(f_input.read())

for file_path in file_list_test_spam:
    with open(file_path,encoding='latin1') as f_input:
        CORPUS_TEST_SPAM.append(f_input.read())

for file_path in file_list_train_ham:
    with open(file_path,encoding='latin1') as f_input:
        CORPUS_TRAIN_HAM.append(f_input.read())

for file_path in file_list_train_spam:
    with open(file_path,encoding='latin1') as f_input:
        CORPUS_TRAIN_SPAM.append(f_input.read())