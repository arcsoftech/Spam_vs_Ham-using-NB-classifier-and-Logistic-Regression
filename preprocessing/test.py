
import string
import pandas as pd
import nltk
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import wordnet
REMOVE_PUNCTUATION_MAP = dict((ord(char), None) for char in string.punctuation)

no_of_docs_ham = 0
no_of_docs_spam = 0


class NB_CLASSIFIER:
    def __init__(self, CORPUS_TRAIN_HAM, CORPUS_TRAIN_SPAM):
        self.CORPUS_TRAIN_SPAM = CORPUS_TRAIN_SPAM
        self.CORPUS_TRAIN_HAM = CORPUS_TRAIN_HAM
    def stop_word_filteration(self,tokens):
        """
        Filter commonly occured stop_words
        """
        stop_words = set(stopwords.words('english'))
        filtered_sentence = [w for w in tokens if w not in stop_words]
        return filtered_sentence
    def tokenize(self,sentence):
        """
        Tokenization
        """
        try:
            tokens = nltk.word_tokenize(
                sentence.lower().translate(REMOVE_PUNCTUATION_MAP))
            filtered_tokens = self.stop_word_filteration(tokens)
            return filtered_tokens
        except Exception as e:
            print("Error Occured",e)
            return []
    def train(self):
        try:
            HAM_TOKEN  = []
            SPAM_TOKEN = []
            for x in self.CORPUS_TRAIN_HAM:
                HAM_TOKEN.append(self.tokenize(x))
            for x in self.CORPUS_TRAIN_SPAM:
                SPAM_TOKEN.append(self.tokenize(x))
            KB= HAM_TOKEN + SPAM_TOKEN
            corpus=[item for sublist in KB for item in sublist]
            vocabulary_features=list(set(corpus))
            corpus_spam=[item for sublist in SPAM_TOKEN for item in sublist]
            corpus_ham=[item for sublist in HAM_TOKEN for item in sublist]
            ham_docs_text=nltk.Text(corpus_ham)
            spam_docs_text=nltk.Text(corpus_spam)

            # ham_docs_text=[nltk.Text(x) for x in HAM_TOKEN]
            # spam_docs_text=[nltk.Text(x) for x in SPAM_TOKEN]
            # word_count_ham=[]
            # word_count_spam=[]
            count_term_ham = {}
            for y in vocabulary_features:
                count_term_ham[y]=ham_docs_text.count(y)
            count_term_spam = {}
            for y in vocabulary_features:
                count_term_spam[y]=spam_docs_text.count(y)
            
            # for x in spam_docs_text:
            #     data = {}
            #     for y in vocabulary_features:
            #         data[y]=x.count(y)
            #     word_count_spam.append(data)
            # ham_data = pd.DataFrame(word_count_ham)
            # spam_data = pd.DataFrame(word_count_spam)
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
    def classify(self,vocabulary,prior,conditional_probability,d):
        token = self.tokenize(d)
        score = {}
        score["spam"]=math.log(prior['spam'])
        score["ham"]=math.log(prior['ham'])
        for t in token:
            if t in vocabulary:
                score["spam"]= score["spam"] + math.log(conditional_probability["spam"][t])
                score["ham"]= score["ham"] + math.log(conditional_probability["ham"][t])
        if(score["spam"]>score["ham"]):
            return "spam"
        elif (score['ham']>score['spam']):
            return "ham"
        else:
            return -1
        

