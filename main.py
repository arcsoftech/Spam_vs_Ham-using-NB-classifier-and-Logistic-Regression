
"""
Name:Arihant Chhajed
Purpose: Machine Learning Class Assignment II
Decription: Spam detection using naive bayesian and logistic regression
"""

from preprocessing import load_data,process_data
import logisticRegression
import naiveBayes
import glob
import os
import sys
import random

from tabulate import tabulate

random.seed(123)
"""
The default testing dataset and training dataset
"""
file_list_test_ham  =  glob.glob(os.path.join(os.getcwd(), "hw2_test","test","ham", "*.txt"))
file_list_test_spam =  glob.glob(os.path.join(os.getcwd(), "hw2_test","test","spam", "*.txt"))
file_list_train_ham  = glob.glob(os.path.join(os.getcwd(), "hw2_train","train","ham", "*.txt"))
file_list_train_spam = glob.glob(os.path.join(os.getcwd(), "hw2_train","train","spam", "*.txt"))

"""
Logistic regression default hyper parametes
"""
learning_rate=0.001
lambda_reg= 0.001
no_of_epoch = 300


print("""
IF you want default paramters then directly pass enter without any arguyment or Please provide the argument in the format as stated below:-
<Ham_Train_Directory_Path> <Spam_Train_Directory_Path> <Ham_Test_Directory_Path> <Spam_Test_Directory_Path> <LearningRate> <No_of_Epoc> <Regularization_Lambda_Paramater>
            """)


if __name__ == "__main__":
    """
    CommandLine Input
    """
    if(sys.argv.__len__() > 1 and sys.argv.__len__() != 8):
        sys.exit("""
            IF you want default paramters then directly pass enter without any arguyment or Please provide the argument in the format as stated below:-
            <Ham_Train_Directory_Path> <Spam_Train_Directory_Path> <Ham_Test_Directory_Path> <Spam_Test_Directory_Path> <LearningRate> <No_of_Epoc> <Regularization_Lambda_Paramater>
            """)
    elif(sys.argv.__len__() > 1):
        try:
            file_list_train_ham = glob.glob(os.path.join(sys.argv[1],"*.txt"))
            file_list_train_spam= glob.glob(os.path.join(sys.argv[2],"*.txt"))
            file_list_test_ham = glob.glob(os.path.join(sys.argv[3],"*.txt"))
            file_list_test_spam= glob.glob(os.path.join(sys.argv[4],"*.txt"))
            learning_rate = float(sys.argv[5])
            lambda_reg = float(sys.argv[7])
            no_of_epoch = int(sys.argv[6])
        except Exception as ex:
            if(type(ex).__name__ == "ValueError"):
                print("Please enter float value for LearningRate and lambda_reg,Integer value for no_of_epoch")
            else:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print (message)
    else:
        print("Default paramters are taken")
    # print(file_list_train_ham,file_list_train_spam,file_list_train_ham,file_list_train_spam,learning_rate,lambda_reg,no_of_epoch)
    
    corpus_train_ham,corpus_train_spam,corpus_test_ham,corpus_test_spam = load_data.initialize(file_list_train_ham,file_list_train_spam,file_list_test_ham,file_list_test_spam)
    
    NB = naiveBayes.NB(corpus_train_ham,corpus_train_spam,corpus_test_ham,corpus_test_spam)
    LR = logisticRegression.LR(corpus_train_ham,corpus_train_spam,corpus_test_ham,corpus_test_spam,learning_rate,lambda_reg,no_of_epoch)
    
    nb_model=NB.train()


    accuracy_nws,accuracy_ham_nws,accuracy_spam_nws = NB.classify(nb_model,False)
    print("Accuracy of dataset using Naive Bayessian classifier without stop word filteration is {0:.2f}%".format(round(accuracy_nws,2)))
    accuracy_n,accuracy_ham_n,accuracy_spam_n = NB.classify(nb_model,True)
    print("Accuracy of dataset using Naive Bayessian classifier with stop word filteration is {0:.2f}%".format(round(accuracy_n,2)))
    lr_model=LR.train(False)
    accuracy_lws,accuracy_ham_lws,accuracy_spam_lws = LR.classify(lr_model,False)
    print("Accuracy of dataset using Logistic Regression classifier without stop word filteration is {0:.2f}%".format(round(accuracy_lws,2)))
    lr_model_stop_words=LR.train(True)
    accuracy_l,accuracy_ham_l,accuracy_spam_l = LR.classify(lr_model_stop_words,True)
    print("Accuracy of dataset using Logictic Regression classifier with stop word filteration is {0:.2f}%".format(round(accuracy_l,2)))
    
    # Optional Section For Report Generation
    # print("Generating report....")
    f = open("output.txt",'w')
    print("NAIVE BAYESIAN")
    print("NAIVE BAYESIAN",file=f)
    print("Overall Accuracy of naive bayesian without stop words filterationis {0:.2f}%".format(round(accuracy_nws,2)),file=f)
    print("Overall Accuracy of naive bayesian with stop words filteration is {0:.2f}%".format(round(accuracy_n,2)),file=f)
    Table=[["False",str(accuracy_ham_nws)+"%",str(accuracy_spam_nws)+"%"],["True",str(accuracy_ham_n)+"%",str(accuracy_spam_n)+"%"]]
    print(tabulate(Table, headers=['StopWords', 'Accuracy(Ham)','Accuracy(Spam)'], tablefmt='orgtbl'))
    print(tabulate(Table, headers=['StopWords',  'Accuracy(Ham)','Accuracy(Spam)'], tablefmt='orgtbl'),file=f)
    print("-----------------------------------------------------------",file=f)
    print("LOGISTIC REGRESSION")
    print("LOGISTIC REGRESSION",file=f)
    print("Overall Accuracy of logistic regression without stop words filterationis {0:.2f}%".format(round(accuracy_lws,2)),file=f)
    print("Overall Accuracy of logistic regression with stop words filteration is {0:.2f}%".format(round(accuracy_l,2)),file=f)
    Table=[["False",str(accuracy_ham_lws)+"%",str(accuracy_spam_lws)+"%"],["True",str(accuracy_ham_l)+"%",str(accuracy_spam_l)+"%"]]
    print(tabulate(Table, headers=['StopWords', 'Accuracy(Ham)','Accuracy(Spam)'], tablefmt='orgtbl'))
    print(tabulate(Table, headers=['StopWords',  'Accuracy(Ham)','Accuracy(Spam)'], tablefmt='orgtbl'),file=f)

    f.close()