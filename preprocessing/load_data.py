
def initialize(HAM_TRAIN_PATH,SPAM_TRAIN_PATH,HAM_TEST_PATH,SPAM_TEST_PATH):
        corpus_test_ham = []
        corpus_test_spam = []
        corpus_train_ham = []
        corpus_train_spam = []


        for file_path in HAM_TEST_PATH:
                with open(file_path,encoding='latin1') as f_input:
                        corpus_test_ham.append(f_input.read())

        for file_path in SPAM_TEST_PATH:
                with open(file_path,encoding='latin1') as f_input:
                        corpus_test_spam.append(f_input.read())

        for file_path in HAM_TRAIN_PATH:
                with open(file_path,encoding='latin1') as f_input:
                        corpus_train_ham.append(f_input.read())

        for file_path in SPAM_TRAIN_PATH:
                with open(file_path,encoding='latin1') as f_input:
                        corpus_train_spam.append(f_input.read())
        
        return corpus_train_ham,corpus_train_spam,corpus_test_ham,corpus_test_spam