
import string
import pandas as pd
import nltk
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import wordnet
REMOVE_PUNCTUATION_MAP = dict((ord(char), None) for char in string.punctuation)

no_of_docs_ham = 0
no_of_docs_spam = 0
def stop_word_filteration(tokens):
    """
    Filter commonly occured stop_words
    """
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in tokens if w not in stop_words]
    return filtered_sentence
def tokenize(sentence,filter=True):
    """
    Tokenization
    """
    try:
        if(filter):
            tokens = nltk.word_tokenize(
                sentence.lower())
            filtered_tokens = stop_word_filteration(tokens)
            return filtered_tokens
        else:
            tokens = nltk.word_tokenize(
                sentence.lower())
            return tokens 
    except Exception as e:
        print("Error Occured",e)
        return []
def logistic_func(z):   
    """Logistic (sigmoid) function
    """
    a = []
    for x in z:
        try:
            denom = ( 1 + math.exp(-x) )
            ans = ( 1 / denom)
            a.append(ans)
        except Exception:
            a.append(0)
    return np.array(a)
def safeLog(z):
    a = []
    for x in z:
        if(x >0):
            a.append(math.log(x))
        else:
            a.append(0)
    a = np.array(a)
    return a
def reg_logLiklihood(x, weights, y, l):
    """Regularizd log-liklihood function (cost function to minimized in logistic
    regression classification with L2 regularization)
    """
    np.seterr(all='raise') 
    z = np.dot(x, weights) 
    z = np.asarray(z,dtype=float)
    reg_term = (l / (2)) * np.dot(weights.T, weights)
    
    error = -1 * np.sum((y * safeLog(logistic_func(z))) + ((1 - y) * safeLog(1-logistic_func(z)))) + reg_term
    return error


class LOGISTIC_CLASSIFIER:
    def __init__(self):
        pass
    def pre_process(self,corpus_ham, corpus_spam,filter,vocabulary = None):
        try:
            HAM_TOKEN  = []
            SPAM_TOKEN = []
            for x in corpus_ham:
                HAM_TOKEN.append(tokenize(x,filter))
            for x in corpus_spam:
                SPAM_TOKEN.append(tokenize(x,filter))
            KB= HAM_TOKEN + SPAM_TOKEN
            corpus=[item for sublist in KB for item in sublist]
            vocabulary_features=list(set(corpus)) if vocabulary is None else vocabulary
            ham_docs_text=[nltk.Text(x) for x in HAM_TOKEN]
            spam_docs_text=[nltk.Text(x) for x in SPAM_TOKEN]
            word_count_spam = []
            word_count_ham = []
            for x in spam_docs_text:
                data = {}
                for y in vocabulary_features:
                    data[y]=x.count(y)
                word_count_spam.append(data)
            for x in ham_docs_text:
                data = {}
                for y in vocabulary_features:
                    data[y]=x.count(y)
                word_count_ham.append(data)
            ham_data = pd.DataFrame(word_count_ham)
            spam_data = pd.DataFrame(word_count_spam)
            ham_data=ham_data.assign(label=0)
            spam_data=spam_data.assign(label=1)
            data=pd.concat([ham_data,spam_data])
            return data
            
        except Exception as e:
            print("error",e)
            return -1
    def train(self,data,learningRate,epoch,l = 0.01):
        X_train = data.values[:, 0:-1] #features vectors
        
        X_train = np.c_[np.ones(np.shape(X_train)[0]), X_train] # for w0 appending extra feature column with value as 1
        # X_train=(X_train-X_train.min())/(X_train.max()-X_train.min()) #normalized data set
        y_train = data.values[:,-1]   #class labels: 0 = ham, 1 = spam
        self.weights = np.zeros(X_train.shape[1]) 
        self.costs = []
        for i in range(epoch):
            z = np.dot(X_train, self.weights)
            errors = y_train - logistic_func(z)
            delta_w = learningRate * (np.dot(errors, X_train) - (l*self.weights))  
            #weight update
            self.weights += delta_w                                
            #Costs
            self.costs.append(reg_logLiklihood(X_train, self.weights, y_train, l))
            self.iterationsPerformed = i
    def classify(self, data):
        """predict class label 
        """ 
        X_test = data.values[:, 0:-1]
        labels = data.values[:,-1]
        print       
        z = self.weights[0] + np.dot(X_test, self.weights[1:])        
        predictions = np.where(z >= 0, 1, 0)
        accuracy=(np.sum(labels == predictions)/predictions.shape[0])*100
        predict_0 = 0
        predict_1 = 0
        for i,v in enumerate(labels):
            if(v == 0 and predictions[i] == v ):
                predict_0 += 1
            if(v == 1 and predictions[i] == v):
                predict_1 += 1
        labels_0 = [x for x in labels if x == 0].__len__()
        labels_1 = [x for x in labels if x ==1].__len__()
        accuracy_ham=(predict_0/labels_0)*100
        accuracy_spam=(predict_1/labels_1)*100
        return accuracy,accuracy_ham,accuracy_spam

class NB_CLASSIFIER:
    def __init__(self, CORPUS_TRAIN_HAM, CORPUS_TRAIN_SPAM):
        self.CORPUS_TRAIN_SPAM = CORPUS_TRAIN_SPAM
        self.CORPUS_TRAIN_HAM = CORPUS_TRAIN_HAM

    def train(self,filter):
        try:
            HAM_TOKEN  = []
            SPAM_TOKEN = []
            for x in self.CORPUS_TRAIN_HAM:
                HAM_TOKEN.append(tokenize(x,filter))
            for x in self.CORPUS_TRAIN_SPAM:
                SPAM_TOKEN.append(tokenize(x,filter))
            KB= HAM_TOKEN + SPAM_TOKEN
            corpus=[item for sublist in KB for item in sublist]
            vocabulary_features=list(set(corpus))
            corpus_spam=[item for sublist in SPAM_TOKEN for item in sublist]
            corpus_ham=[item for sublist in HAM_TOKEN for item in sublist]
            ham_docs_text=nltk.Text(corpus_ham)
            spam_docs_text=nltk.Text(corpus_spam)
            count_term_ham = {}
            for y in vocabulary_features:
                count_term_ham[y]=ham_docs_text.count(y)
            count_term_spam = {}
            for y in vocabulary_features:
                count_term_spam[y]=spam_docs_text.count(y)
            no_of_docs_ham=HAM_TOKEN.__len__()
            no_of_docs_spam=SPAM_TOKEN.__len__()
            prior={}
            prior["ham"]= no_of_docs_ham/(no_of_docs_ham+no_of_docs_spam)
            prior["spam"]= no_of_docs_spam/(no_of_docs_ham+no_of_docs_spam)
            conditional_probability = {'spam':{},'ham':{}}
            t_ham_dash = sum([x+1 for x in count_term_ham.values()])
            t_spam_dash = sum([x+1 for x in count_term_spam.values()])
            for t in vocabulary_features:
                conditional_probability['spam'][t]=((count_term_spam[t]+1)/t_spam_dash)
                conditional_probability['ham'][t]=((count_term_ham[t]+1)/t_ham_dash)
            return vocabulary_features,prior,conditional_probability
        except Exception as e:
            print("error",e)
            return 0,0,0
    def classify(self,filter,vocabulary,prior,conditional_probability,d):
        token = tokenize(d)
        score = {}
        score["spam"]=math.log(prior['spam'])
        score["ham"]=math.log(prior['ham'])
        for t in token:
            if t in vocabulary:
                score["spam"]= score["spam"] + math.log(conditional_probability["spam"][t])
                score["ham"]= score["ham"] + math.log(conditional_probability["ham"][t])
        if(score["spam"]>score["ham"]):
            return 0
        elif (score['ham']>score['spam']):
            return 1
        else:
            return -1
