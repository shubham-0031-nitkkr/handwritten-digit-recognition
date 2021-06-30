import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def checkAccuracy(test_X, test_Y, clf):
    predict_Y = clf.predict(test_X)
    print("Accuracy of our model is ", 100*accuracy_score(test_Y, predict_Y))

def predictValue(test_X, test_Y, clf):
    val = int(input("Enter any value between (1 to 10) for demo of our model : "))
    print("Your handwritten digit image is recognizing... ")
    digit = test_X[val]
    digit_display = digit.reshape(28, 28)
    plt.imshow(digit_display, interpolation = "nearest")
    plt.show()
    print("Digit is : ", clf.predict([test_X[val]]))
    isCheckAccuracy = int(input("Press 1 for checking accuracy of our model else Press 0 : "))
    if isCheckAccuracy == 1:
        checkAccuracy(test_X, test_Y, clf)
    else:
        print("Thank You for using our model !")


def selectClassifier(train_X, train_Y):
    print("Now, Select any one given classifier ")
    print("Press 1 for Support Vector Machine or Press 2 for K-nearest Neighbour or Press 3 for Logistic Regression : ")
    clacfyr = int(input())
    if clacfyr == 1:
        clf = SVC()
        clf.fit(train_X, train_Y)
    elif clacfyr == 2:
        clf = KNeighborsClassifier()
        clf.fit(train_X, train_Y)
    elif clacfyr == 3:
        clf = LogisticRegression(solver='lbfgs', max_iter=7000)
        clf.fit(train_X, train_Y)
    return clf


def fetchData():
    print("Dataset is fetching from MNIST dataset...")
    mnist = fetch_openml('mnist_784')
    print("Dataset has been fetched !")
    print("Dataset is spliting in training and testing data...")
    x = mnist['data']
    y = mnist['target']
    train_X = x[:6000]                          # training data
    train_Y = y[:6000]                          # training data
    data_shuffling = np.random.permutation(6000)
    train_X = train_X[data_shuffling]
    train_Y = train_Y[data_shuffling]
    test_X = x[6000:7000]                       # testing data
    test_Y = y[6000:7000]                       # testing data
    print("Dataset has been split !")
    clf = selectClassifier(train_X, train_Y)
    predictValue(test_X, test_Y, clf)

def main():
    print("Hello, this is Handwritten Digit Recognition project : ")
    print("Press Enter to continue ")
    input()
    fetchData()

if True:
    main()
