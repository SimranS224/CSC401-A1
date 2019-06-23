from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

import argparse
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from scipy import stats
from sklearn.svm import LinearSVC
import csv
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from scipy import stats

# Classifers
myuniquesvclassifier = svm.LinearSVC()
myuniuqe2svclassifier = svm.SVC(kernel='rbf', gamma=2, max_iter=1200)
myuniquerf = RandomForestClassifier(max_depth=10, n_estimators=10)
myuniqueclf = MLPClassifier(alpha=0.05)
myuniqueabc = AdaBoostClassifier()
# the other liwc features
liwcfeatsunique1 = [line.rstrip() for line in open("/u/cs401/A1/feats/feats.txt", "r")]


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return (C.diagonal().sum() / C.sum())


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    arraytoreturn = [C[i][i] / sum(C[i]) for i in range(len(C))]
    return arraytoreturn


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    arraytoreturn = [C[i][i] / sum(C[..., i]) for i in range(len(C))]
    return arraytoreturn


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    # load the file and convert it to a numpy array
    filen = np.load(filename)
    file = filen[filen.files[0]]
    # get all columns up to the 173rd
    X = file[..., :173]
    # get 173rd column
    y = file[..., 173]
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # LINEARSVM
    myuniquesvclassifier.fit(X_train, y_train)
    y_pred = myuniquesvclassifier.predict(X_test)
    c = confusion_matrix(y_test, y_pred)

    confuslinearsvc = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            confuslinearsvc.append(c[i][j])
    # format the output of LINEAR SVM
    outputarraylinearsvc = [1, accuracy(c), recall(c)[0], recall(c)[1], recall(c)[2], recall(c)[3], precision(c)[0],
                            precision(c)[1], precision(c)[2], precision(c)[3]]
    # add confustion matrix to the output of LINEAR SVM
    outputarraylinearsvc.extend(confuslinearsvc)

    # RADIAL SVM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    myuniuqe2svclassifier.fit(X_train, y_train)
    newy_pred = myuniuqe2svclassifier.predict(X_test)
    c1 = confusion_matrix(y_test, newy_pred)

    confussvc = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            confussvc.append(c[i][j])
    # format the output of RADIAL SVM
    outputarraysvc = [2, accuracy(c1), recall(c1)[0], recall(c1)[1], recall(c1)[2], recall(c1)[3], precision(c1)[0],
                      precision(c1)[1], precision(c1)[2], precision(c1)[3]]
    # add confustion matrix to the output of RADIAL SVM
    outputarraysvc.extend(confussvc)

    # RandomForestClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    myuniquerf.fit(X_train, y_train);
    rfpredictions10 = myuniquerf.predict(X_test)
    c2 = confusion_matrix(y_test, rfpredictions10)

    confusforest = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            confusforest.append(c[i][j])
    # format the output of RandomForestClassifier
    outputarrayforest = [3, accuracy(c2), recall(c2)[0], recall(c2)[1], recall(c2)[2], recall(c2)[3], precision(c2)[0],
                         precision(c2)[1], precision(c2)[2], precision(c2)[3]]
    # add confustion matrix to the output of RandomForestClassifier
    outputarrayforest.extend(confusforest)

    # MLP classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    myuniqueclf.fit(X_train, y_train)
    mlppredict = myuniqueclf.predict(X_test)
    c4 = confusion_matrix(y_test, mlppredict)
    confusmlp = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            confusmlp.append(c[i][j])
    # format the output of MLP classifier
    outputmlp = [4, accuracy(c4), recall(c4)[0], recall(c4)[1], recall(c4)[2], recall(c4)[3], precision(c4)[0],
                 precision(c4)[1], precision(c4)[2], precision(c4)[3]]
    # add confustion matrix to the output of MLP classifier
    outputmlp.extend(confusmlp)

    # AdaBoostClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    myuniqueabc.fit(X_train, y_train)
    y_predabc = myuniqueabc.predict(X_test)
    c5 = confusion_matrix(y_test, y_predabc)
    confusadaboost = []
    for i in range(len(c)):
        for j in range(len(c[i])):
            confusadaboost.append(c[i][j])
    # format the output of AdaBoostClassifier
    outputadaboost = [5, accuracy(c5), recall(c5)[0], recall(c5)[1], recall(c5)[2], recall(c5)[3], precision(c5)[0],
                      precision(c5)[1], precision(c5)[2], precision(c5)[3]]
    # add confustion matrix to the output of AdaBoostClassifier
    outputadaboost.extend(confusadaboost)

    output = [outputarraylinearsvc, outputarraysvc, outputarrayforest, outputmlp, outputadaboost]

    # write to the csv , each element is a list which is displayed on each row

    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    with open('a1_3.1final.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in output:
            writer.writerow(row)

    f.close()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    ibest = 0
    themax = 0
    for x in output:
        currmax = x[1]
        if currmax > themax:
            themax = x[1]
            ibest = x[0]

    return (X_train, X_test, y_train, y_test, ibest)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    # all classifers
    classifers_list = [myuniquesvclassifier, myuniuqe2svclassifier, myuniquerf, myuniqueclf, myuniqueabc]
    # the best one
    theclassifertouse = classifers_list[iBest - 1]

    # 1k data set
    x_train1k = X_train[:1000]
    x_test1k = X_test
    y_train1k = y_train[:1000]
    y_test1k = y_test
    theclassifertouse.fit(x_train1k, y_train1k)
    y_predabc = theclassifertouse.predict(x_test1k)
    c = confusion_matrix(y_test1k, y_predabc)

    # 2k data set
    x_train2k = X_train[:2000]
    x_test2k = X_test
    y_train2k = y_train[:2000]
    y_test2k = y_test

    theclassifertouse.fit(x_train2k, y_train2k)
    y_predabc = theclassifertouse.predict(x_test2k)
    c1 = confusion_matrix(y_test2k, y_predabc)

    # 5k data set
    x_train5k = X_train[:5000]
    x_test5k = X_test
    y_train5k = y_train[:5000]
    y_test5k = y_test

    theclassifertouse.fit(x_train5k, y_train5k)
    y_predabc = theclassifertouse.predict(x_test5k)
    c2 = confusion_matrix(y_test5k, y_predabc)

    # 10k data set
    x_train10k = X_train[:10000]
    x_test10k = X_test
    y_train10k = y_train[:10000]
    y_test10k = y_test
    theclassifertouse.fit(x_train10k, y_train10k)
    y_predabc = theclassifertouse.predict(x_test10k)
    c3 = confusion_matrix(y_test10k, y_predabc)

    # 20k data set
    x_train20k = X_train[:20000]
    x_test20k = X_test
    y_train20k = y_train[:20000]
    y_test20k = y_test
    theclassifertouse.fit(x_train20k, y_train20k)
    y_predabc = theclassifertouse.predict(x_test20k)
    c4 = confusion_matrix(y_test20k, y_predabc)
    # format output to be outputted into csv
    outputadaboost = [[accuracy(c), accuracy(c1), accuracy(c2), accuracy(c3), accuracy(c4)], [
        "We expect the accuracy to increase as the number of training sample increases. This is because of the way Adaboost works, since it keeps assigning higher weight to the wrong classified observations on each iteration , and since it keeps repeating this until the it has reached the maximum number of estimators, we can argue that if there is more data it will be more accurate. But with one caveat due to the fact that Adaboost is sensitive to noise - if the data we have has a lot of noise it can be affected by outliers. Similarly, other classifiers can also become more accurate with more data, however there are two reasons why more data might not improve accuracy and they are if the model is too simple resulting in high bias or if the model is too complex i.e uses too many features. However in this case more data and fewer features will resolve this problem."]]
    # write to the csv , each element is a list which is displayed on each row

    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    with open('a1_3.2final.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in outputadaboost:
            writer.writerow(row)

    f.close()

    return (x_train1k, y_train1k)


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''

    thebest = i
    listtooutput = []
    # the  29 features
    features = ["first_person_pronouns", "second_person_pronouns", "third_person_pronouns",
                "coordinating_conjunctions", "past_tense_verbs", "future_tense_verbs",
                "commas", "multi_character_punctuation_tokens", "common_nouns", "proper_nouns",
                "adverbs", "wh_words", "slang_acronyms", "words_in_uppercase",
                "Avg_length_of_sentences", "Avg_length_of_tokensexp", "numsentences",
                "Avg_of_AoA", "Avg_of_IMG", "Avg_of_FAM", "sdofAoA", "stdofIMG",
                "stdofFAM", "avgofvmeansum", "avgofamean", "avgofdmean", "stdvmeansum", "stdameansum",
                "stddmeansum"]
    # the liwc features from the document
    features.extend(liwcfeatsunique1)

    kvalues = [5, 10, 20, 30, 40, 50]

    onekfeats = []
    onekpvalues = []
    theonekfeats = []
    size1kbest5kfeats = []
    size1kbest5kfeatsindexes = []
    for kv in kvalues:
        selector = SelectKBest(f_classif, kv)
        X_new = selector.fit_transform(X_1k, y_1k)
        pp = selector.pvalues_
        thecurrentfeats = selector.get_support()
        indexes = []
        for i in range(len(thecurrentfeats)):
            if thecurrentfeats[i]:
                indexes.append(i)
        # indexes = [int(i) for i in pp[the32feats]]
        listoffeats = []
        for x in indexes:
            listoffeats.append(features[x])
        #  the amount of features
        arraytoreturn = [len(listoffeats)]
        # the array of pp values
        arraytoreturn.extend(pp[thecurrentfeats])
        # the pp values
        onekpvalues.extend([pp[thecurrentfeats]])
        # add the formatted array to the final output of the 1k data set size feats for comapring with 32k data set
        onekfeats.append(arraytoreturn)
        theonekfeats.append(listoffeats)
        # when k - 5 add it to a special list for comparison later on in with the 32k data set size results
        if kv == 5:
            size1kbest5kfeatsindexes.extend(indexes)
            size1kbest5kfeats.extend(listoffeats)

    size32kbest5kfeats = []
    size32kbest5kfeatsindexes = []
    x_train32k = X_train[:32000]
    x_test32k = X_test
    y_train32k = y_train[:32000]
    y_test32k = y_test
    the32kfeats = []
    t32kpvalues = []
    for kv in kvalues:
        selector = SelectKBest(f_classif, kv)
        X_new = selector.fit_transform(x_train32k, y_train32k)

        pp = selector.pvalues_

        thecurrentfeats = selector.get_support()

        indexes = []
        for i in range(len(thecurrentfeats)):
            if thecurrentfeats[i]:
                indexes.append(i)
        listoffeats = []
        for x in indexes:
            listoffeats.append(features[x])
        #  the amount of feature
        arraytoreturn = [len(listoffeats)]
        # the array of pp values
        arraytoreturn.extend(pp[thecurrentfeats])
        # add to the final array to output the 32k feats in the csv
        listtooutput.append(arraytoreturn)
        # add the formatted array to the final output of the 32k data set size feats
        the32kfeats.append(listoffeats)
        t32kpvalues.extend(pp[thecurrentfeats])
        # the 5 best features of the 32k data set size
        if kv == 5:
            size32kbest5kfeatsindexes.extend(indexes)
            size32kbest5kfeats.extend(listoffeats)


    classifers_list = [myuniquesvclassifier, myuniuqe2svclassifier, myuniquerf, myuniqueclf, myuniqueabc]
    theclassifertouse = classifers_list[(thebest - 1)]
    # make a new numpy array with the 5 best features
    xtrain1k = np.zeros([len(X_1k), 5], dtype=float)
    for i in range(len(X_1k)):
        xtrain1k[i] = X_1k[i][size1kbest5kfeatsindexes[0]]

    xtest1k = np.zeros([len(X_test), 5], dtype=float)
    for i in range(len(X_test)):
        xtest1k[i] = X_test[i][size1kbest5kfeatsindexes[0]]
    # train it on AdaBoost
    theclassifertouse.fit(xtrain1k, y_1k)
    y_predabc = theclassifertouse.predict(xtest1k)
    c = confusion_matrix(y_test, y_predabc)

    # 32k class
    # make a new numpy array with the 5 best features
    xtrain32k = np.zeros([len(x_train32k), 5], dtype=float)
    for i in range(len(x_train32k)):
        xtrain32k[i] = x_train32k[i][size32kbest5kfeatsindexes[0]]

    xtest32k = np.zeros([len(x_test32k), 5], dtype=float)
    for i in range(len(x_test32k)):
        xtest32k[i] = x_test32k[i][size32kbest5kfeatsindexes[0]]
    # train it on AdaBoost
    theclassifertouse.fit(xtrain32k, y_train32k)
    y_predabc = theclassifertouse.predict(xtest32k)
    c1 = confusion_matrix(y_test, y_predabc)
    # add the accuracies to a list which will be added to the last line of the csv
    accuracies = [accuracy(c), accuracy(c1)]
    listtooutput.append(accuracies)
    # write to the csv , each element is a list which is displayed on each row
    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    with open('a1_3.3final.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in listtooutput:
            writer.writerow(row)

    f.close()
    # testing code for checking values
    output = []
    for i in range(len(theonekfeats)):
        output.append(list(set(theonekfeats[i]).intersection(the32kfeats[i])))
        print(list(set(theonekfeats[i]).intersection(the32kfeats[i])))
    print(output)
    print("the onekpvalues \n")
    # print(onekpvalues)
    print("\n")
    output1 = []
    # o1 = list(np.array(onekpvalues).tolist())
    # o2 = list(np.array(t32kpvalues).tolist())
    for i in range(len(theonekfeats)):
        print(onekpvalues[i])

    print("the 32 \n")
    for i in range(len(t32kpvalues)):
        print(t32kpvalues[i])
    print(size32kbest5kfeats)


def class34(filename, i):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    classifers_list = [myuniquesvclassifier, myuniuqe2svclassifier, myuniquerf, myuniqueclf, myuniqueabc]
    # subtract 1 since the rank values start at 1
    thebest = i - 1
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    filen = np.load(filename)
    file = filen[filen.files[0]]
    X = file[..., :173]
    y = file[..., 173]
    accuracies = []

    # iterate for 5 folds
    for train_index, test_index in kf.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        count = 0
        currarray = []
        # use each classifer on the data set once per fold
        for classifer in classifers_list:
            classifer.fit(X_train, y_train);

            thepred = classifer.predict(X_test)
            c = confusion_matrix(y_test, thepred)
            theaccuracy = accuracy(c)
            currarray.append(theaccuracy)

        accuracies.append(currarray)
    # the accuracies of each individual classifer
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    a5 = []
    # index the prevousuly created list to get the column values
    print(accuracies)
    for i in range(5):
        a1.append(accuracies[i][0])
        a2.append(accuracies[i][1])
        a3.append(accuracies[i][2])
        a4.append(accuracies[i][3])
        a5.append(accuracies[i][4])
    all = [a1, a2, a3, a4, a5]
    # get the best list
    thebestlistofa = all[thebest]
    thefour = []
    for x in all:
        if x != thebestlistofa:
            thefour.append(x)
    print(thefour)
    print(len(thefour))
    # compare the best classifer with the other four classifers to get the significance of the performance
    # of the best one
    compare1 = stats.ttest_rel(thebestlistofa, thefour[0])[1]
    compare2 = stats.ttest_rel(thebestlistofa, thefour[1])[1]
    compare3 = stats.ttest_rel(thebestlistofa, thefour[2])[1]
    compare4 = stats.ttest_rel(thebestlistofa, thefour[3])[1]
    pvalueslist = [compare1, compare2, compare3, compare4]
    listtooutput = [accuracies[0], accuracies[1], accuracies[2], accuracies[3], accuracies[4], pvalueslist]

    # write to the csv , each element is a list which is displayed on each row
    csv.register_dialect('myDialect',
                         quoting=csv.QUOTE_ALL,
                         skipinitialspace=True)

    with open('a1_3.4final.csv', 'w') as f:
        writer = csv.writer(f, dialect='myDialect')
        for row in listtooutput:
            writer.writerow(row)

    f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)

    args = parser.parse_args()

    x_train, x_test, y_train, y_test, i_best = class31(args.input)
    x_1k, y_1k = class32(x_train, x_test, y_train, y_test, i_best)
    # class33(x_train, x_test, y_train, y_test, i_best, x_1k, y_1k)
    # file = myfile1["arr_0"]
    # print(file.shape)
    # X = file[..., :173]
    # y = file[..., 173]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    # x_1k, y_1k = class32(X_train, X_test, y_train, y_test, 5)
    class33(x_train, x_test, y_train, y_test, i_best, x_1k, y_1k)
    class34(args.input, i_best)

    # TODO : complete each classification experiment, in sequence.
