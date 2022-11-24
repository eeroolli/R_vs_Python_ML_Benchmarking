# https://jooskorstanje.com/speed-benchmark-python.html


import pandas as pd
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

## Data

names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=names)
iris.head()

# split train test
train, test = train_test_split(iris, test_size=0.2)

X_train = train.drop('class', axis=1)
y_train = train['class']

X_test = test.drop('class', axis=1)
y_test = test['class']

## Fitting the Models

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# KNN
parameters = {'n_neighbors': range(1, 10)}
knn = GridSearchCV(KNeighborsClassifier(), parameters, scoring='accuracy', n_jobs=3, cv=KFold(n_splits=5))
knn.fit(X_train, y_train)

# SVM
parameters = {'C': range(1, 11)}
svc = GridSearchCV(svm.SVC(kernel='linear'), parameters, scoring='accuracy', n_jobs=3, cv=KFold(n_splits=5))
svc.fit(X_train, y_train)

## Evaluate
lr_test_acc = lr.score(X_test, y_test)
lda_test_acc = lda.score(X_test, y_test)
knn_test_acc = knn.best_estimator_.score(X_test, y_test)
svc_test_acc = svc.best_estimator_.score(X_test, y_test)

print(lr_test_acc, lda_test_acc, knn_test_acc, svc_test_acc)


# Speed Benchmark

def main():
    names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names=names)

    train, test = train_test_split(iris, test_size=0.2)
    X_train = train.drop('class', axis=1)
    y_train = train['class']
    X_test = test.drop('class', axis=1)
    y_test = test['class']

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    parameters = {'n_neighbors': range(1, 10)}
    knn = GridSearchCV(KNeighborsClassifier(), parameters, scoring='accuracy', n_jobs=3, cv=KFold(n_splits=5))
    knn.fit(X_train, y_train)

    parameters = {'C': range(1, 11)}
    svc = GridSearchCV(svm.SVC(kernel='linear'), parameters, scoring='accuracy', n_jobs=3, cv=KFold(n_splits=5))
    svc.fit(X_train, y_train)

    lr_test_acc = lr.score(X_test, y_test)
    lda_test_acc = lda.score(X_test, y_test)
    knn_test_acc = knn.best_estimator_.score(X_test, y_test)
    svc_test_acc = svc.best_estimator_.score(X_test, y_test)

    print(lr_test_acc, lda_test_acc, knn_test_acc, svc_test_acc)


from datetime import datetime as dt

now = dt.now()

for i in range(100):
    print(i)
    main()

now2 = dt.now()
print(now2 - now)

print(f"{((now2 - now) / 100).seconds + ((now2 - now) / 100).microseconds / 1000000} seconds per iteration.")

## Total time was 1:36, and time per iteration was 0.97s. Which is 5 times faster than R.
