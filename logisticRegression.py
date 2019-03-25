from preprocessing import load_data,process_data

from preprocessing import process_data

class LR:
    def __init__(self,CORPUS_TRAIN_HAM,CORPUS_TRAIN_SPAM,CORPUS_TEST_HAM,CORPUS_TEST_SPAM,LEARNING_RATE,LAMBDA,NO_OF_EPOC):
        self.CORPUS_TRAIN_HAM=CORPUS_TRAIN_HAM
        self.CORPUS_TRAIN_SPAM=CORPUS_TRAIN_SPAM
        self.CORPUS_TEST_HAM=CORPUS_TEST_HAM
        self.CORPUS_TEST_SPAM=CORPUS_TEST_SPAM
        self.LEARNING_RATE = LEARNING_RATE
        self.LAMBDA= LAMBDA
        self.NO_OF_EPOC = NO_OF_EPOC
    def train(self,stop_words):
        lc = process_data.LOGISTIC_CLASSIFIER()
        train_data = lc.pre_process(self.CORPUS_TRAIN_HAM,self.CORPUS_TRAIN_SPAM,stop_words)
        self.vocabulary = train_data.columns[0:-1]
        lc.train(train_data,self.LEARNING_RATE,self.NO_OF_EPOC,self.LAMBDA)
        return lc
    def classify(self,lc,stop_words):
        test_data = lc.pre_process(self.CORPUS_TEST_HAM,self.CORPUS_TEST_SPAM,stop_words,self.vocabulary)
        return lc.classify(test_data)
   
