from preprocessing import process_data

class NB:
    def __init__(self,CORPUS_TRAIN_HAM,CORPUS_TRAIN_SPAM,CORPUS_TEST_HAM,CORPUS_TEST_SPAM):
        self.CORPUS_TRAIN_HAM=CORPUS_TRAIN_HAM
        self.CORPUS_TRAIN_SPAM=CORPUS_TRAIN_SPAM
        self.CORPUS_TEST_HAM=CORPUS_TEST_HAM
        self.CORPUS_TEST_SPAM=CORPUS_TEST_SPAM
    def train(self):
        return process_data.NB_CLASSIFIER(self.CORPUS_TRAIN_HAM,self.CORPUS_TRAIN_SPAM)
    def classify(self,train_model,stop_words):
        VOCAB,PRIOR,CONDITIONAL_PROBABILITY = train_model.train(stop_words)
        correct_classify_ham=0
        correct_classify_spam=0
        incorrect_classify_ham=0
        incorrect_classify_spam=0
        correct_classify=0
        incorrect_classify=0
        for d in self.CORPUS_TEST_HAM:
            class_name = train_model.classify(stop_words,VOCAB,PRIOR,CONDITIONAL_PROBABILITY,d)
            if(class_name == 1):
                correct_classify_ham = correct_classify_ham + 1
            else:
                incorrect_classify_ham = incorrect_classify_ham + 1
        for d in self.CORPUS_TEST_SPAM:
            class_name = train_model.classify(stop_words,VOCAB,PRIOR,CONDITIONAL_PROBABILITY,d)
            if(class_name == 0):
                correct_classify_spam = correct_classify_spam + 1
            else:
                incorrect_classify_spam = incorrect_classify_spam + 1
        correct_classify = correct_classify_ham + correct_classify_spam
        incorrect_classify = incorrect_classify_ham + incorrect_classify_spam
        accuracy = (correct_classify/(correct_classify+incorrect_classify)) * 100
        accuracy_ham = (correct_classify_ham/self.CORPUS_TEST_HAM.__len__()) * 100
        accuracy_spam = (correct_classify_spam/self.CORPUS_TEST_SPAM.__len__()) * 100
        return accuracy,accuracy_ham,accuracy_spam

