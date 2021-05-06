import io

import flask

# linear algebra
import numpy as np

# data processing
import pandas as pd

# data visualization

import seaborn as sns
from flask import jsonify, make_response, render_template

from matplotlib import pyplot as plt
from matplotlib import style
from sklearn import metrics

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import statsmodels.api as sm
from sklearn.feature_selection import RFE

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

app = flask.Flask(__name__, static_url_path='')


# app.config["DEBUG"] = True


def toText(s):
    response = make_response(s, 200)
    response.mimetype = "text/plain"
    return response


@app.route('/step1_2', methods=['GET'])
def step1_2():
    train = pd.read_csv("train.csv")
    buf = io.StringIO()
    # train.head().to_string
    s = train.head().to_string()
    return toText(s)


@app.route('/step2_3', methods=['GET'])
def step2_3():
    train = pd.read_csv("train.csv")
    s = '(' + str(train.shape[0]) + ',' + str(train.shape[1]) + ')'
    return toText(s)


@app.route('/step2_4', methods=['GET'])
def step2_4():
    train = pd.read_csv("train.csv")
    buf = io.StringIO()
    train.info(buf=buf)
    s = train.head().to_string()
    return toText(s)


@app.route('/step2_5', methods=['GET'])
def step2_5():
    train = pd.read_csv("train.csv")
    return toText(train.describe().to_string())


@app.route('/step3_6-7', methods=['GET'])
def step3_6_7():
    train = pd.read_csv("train.csv")
    train = train.drop('Cabin', axis=1)
    return toText(train.isnull().sum().to_string())


@app.route('/step4_13-14', methods=['GET'])
def step4_13_14():
    train = pd.read_csv("train.csv")
    # drpping the Parch as it is having low correlation with survived(target variable)
    train = train.drop('Parch', axis=1)
    # dropping ticket and Parch as it is not required for further analysis
    train = train.drop('Ticket', axis=1)
    train = train.drop('Name', axis=1)
    # importing the test data set and checking
    test = pd.read_csv("test.csv")
    return toText(test.head().to_string())


@app.route('/step4_15-17', methods=['GET'])
def step4_15_17():
    train = pd.read_csv("train.csv")
    # drpping the Parch as it is having low correlation with survived(target variable)
    train = train.drop('Parch', axis=1)
    # dropping ticket and Parch as it is not required for further analysis
    train = train.drop('Ticket', axis=1)
    train = train.drop('Name', axis=1)
    # importing the test data set and checking
    test = pd.read_csv("test.csv")

    # converting  the categorical variables to numeric as required for model building

    dummy1 = pd.get_dummies(train[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    train = pd.concat([train, dummy1], axis=1)

    dummy1 = pd.get_dummies(test[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    test = pd.concat([test, dummy1], axis=1)

    # since the dummies are already present
    train = train.drop(['Sex', 'Embarked'], 1)
    test = test.drop(['Sex', 'Embarked'], 1)

    return toText(train.head().to_string())


@app.route('/step_6', methods=['GET'])
def step_6():
    train = pd.read_csv("train.csv")
    train.head()
    train.shape
    train.info()
    train.describe()
    train = train.drop('Cabin', axis=1)
    train.isnull().sum()
    # taking only not null values from 'Age' and 'Embarked' since they have not null values
    train = train[train['Age'].notnull()]
    train = train[train['Embarked'].notnull()]
    train.isnull().sum()

    # drpping the Parch as it is having low correlation with survived(target variable)
    train = train.drop('Parch', axis=1)
    # dropping ticket and Parch as it is not required for further analysis
    train = train.drop('Ticket', axis=1)
    train = train.drop('Name', axis=1)
    # importing the test data set and checking
    test = pd.read_csv("test.csv")
    test.head()
    # converting  the categorical variables to numeric as required for model building

    dummy1 = pd.get_dummies(train[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    train = pd.concat([train, dummy1], axis=1)

    dummy1 = pd.get_dummies(test[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    test = pd.concat([test, dummy1], axis=1)

    # since the dummies are already present
    train = train.drop(['Sex', 'Embarked'], 1)
    test = test.drop(['Sex', 'Embarked'], 1)
    train.head()
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
    X_test = test

    logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())

    return toText(logm1.fit().summary().as_text())


#
# @app.route('/step_7_22-27', methods=['GET'])
# def step_7_22_27():
#     train = pd.read_csv("train.csv")
#     train.head()
#     train.shape
#     train.info()
#     train.describe()
#     train = train.drop('Cabin', axis=1)
#     train.isnull().sum()
#     # taking only not null values from 'Age' and 'Embarked' since they have not null values
#     train = train[train['Age'].notnull()]
#     train = train[train['Embarked'].notnull()]
#     train.isnull().sum()
#
#     # drpping the Parch as it is having low correlation with survived(target variable)
#     train = train.drop('Parch', axis=1)
#     # dropping ticket and Parch as it is not required for further analysis
#     train = train.drop('Ticket', axis=1)
#     train = train.drop('Name', axis=1)
#     # importing the test data set and checking
#     test = pd.read_csv("test.csv")
#     test.head()
#     # converting  the categorical variables to numeric as required for model building
#
#     dummy1 = pd.get_dummies(train[['Sex', 'Embarked']], drop_first=True)
#     # Adding the results to the master dataframe
#     train = pd.concat([train, dummy1], axis=1)
#
#     dummy1 = pd.get_dummies(test[['Sex', 'Embarked']], drop_first=True)
#     # Adding the results to the master dataframe
#     test = pd.concat([test, dummy1], axis=1)
#
#     # since the dummies are already present
#     train = train.drop(['Sex', 'Embarked'], 1)
#     test = test.drop(['Sex', 'Embarked'], 1)
#     train.head()
#     X_train = train.drop('Survived', axis=1)
#     y_train = train['Survived']
#     X_test = test
#
#     logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())
#
#     logreg = LogisticRegression()
#     rfe = RFE(logreg, 6)  # running RFE with 13 variables as output
#     rfe = rfe.fit(X_train, y_train)
#     rfe.support_
#     list(zip(X_train.columns, rfe.support_, rfe.ranking_))
#     col = X_train.columns[rfe.support_]
#     X_train.columns[~rfe.support_]
#     X_train_sm = sm.add_constant(X_train[col])
#     logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
#     res = logm2.fit()
#
#     return toText(res.summary().as_text())


@app.route('/step_7', methods=['GET'])
def step_7():
    train = pd.read_csv("train.csv")
    train.head()
    train.shape
    train.info()
    train.describe()
    train = train.drop('Cabin', axis=1)
    train.isnull().sum()
    # taking only not null values from 'Age' and 'Embarked' since they have not null values
    train = train[train['Age'].notnull()]
    train = train[train['Embarked'].notnull()]
    train.isnull().sum()

    # drpping the Parch as it is having low correlation with survived(target variable)
    train = train.drop('Parch', axis=1)
    # dropping ticket and Parch as it is not required for further analysis
    train = train.drop('Ticket', axis=1)
    train = train.drop('Name', axis=1)
    # importing the test data set and checking
    test = pd.read_csv("test.csv")
    test.head()
    # converting  the categorical variables to numeric as required for model building

    dummy1 = pd.get_dummies(train[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    train = pd.concat([train, dummy1], axis=1)

    dummy1 = pd.get_dummies(test[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    test = pd.concat([test, dummy1], axis=1)

    # since the dummies are already present
    train = train.drop(['Sex', 'Embarked'], 1)
    test = test.drop(['Sex', 'Embarked'], 1)
    train.head()
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
    X_test = test

    logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())

    logreg = LogisticRegression()
    rfe = RFE(logreg, 6)  # running RFE with 13 variables as output
    rfe = rfe.fit(X_train, y_train)
    rfe.support_
    list(zip(X_train.columns, rfe.support_, rfe.ranking_))
    col = X_train.columns[rfe.support_]
    X_train.columns[~rfe.support_]
    X_train_sm = sm.add_constant(X_train[col])
    logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
    res = logm2.fit()

    # Getting the predicted values on the train set
    y_train_pred = res.predict(X_train_sm)
    y_train_pred = y_train_pred.values.reshape(-1)

    y_train_pred_final = pd.DataFrame({'Survived': y_train.values, 'Survived_prob': y_train_pred})
    y_train_pred_final['PassengerId'] = y_train.index
    y_train_pred_final.head()

    y_train_pred_final['predicted'] = y_train_pred_final.Survived_prob.map(lambda x: 1 if x > 0.54 else 0)

    # Let's see the head
    # Let's check the overall accuracy.
    return toText(str(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted)))


@app.route('/step_8', methods=['GET'])
def step_8():
    train = pd.read_csv("train.csv")
    train.head()
    train.shape
    train.info()
    train.describe()
    train = train.drop('Cabin', axis=1)
    train.isnull().sum()
    # taking only not null values from 'Age' and 'Embarked' since they have not null values
    train = train[train['Age'].notnull()]
    train = train[train['Embarked'].notnull()]
    train.isnull().sum()

    # drpping the Parch as it is having low correlation with survived(target variable)
    train = train.drop('Parch', axis=1)
    # dropping ticket and Parch as it is not required for further analysis
    train = train.drop('Ticket', axis=1)
    train = train.drop('Name', axis=1)
    # importing the test data set and checking
    test = pd.read_csv("test.csv")
    test.head()
    # converting  the categorical variables to numeric as required for model building

    dummy1 = pd.get_dummies(train[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    train = pd.concat([train, dummy1], axis=1)

    dummy1 = pd.get_dummies(test[['Sex', 'Embarked']], drop_first=True)
    # Adding the results to the master dataframe
    test = pd.concat([test, dummy1], axis=1)

    # since the dummies are already present
    train = train.drop(['Sex', 'Embarked'], 1)
    test = test.drop(['Sex', 'Embarked'], 1)
    train.head()
    X_train = train.drop('Survived', axis=1)
    y_train = train['Survived']
    X_test = test

    logm1 = sm.GLM(y_train, (sm.add_constant(X_train)), family=sm.families.Binomial())

    logreg = LogisticRegression()
    rfe = RFE(logreg, 6)  # running RFE with 13 variables as output
    rfe = rfe.fit(X_train, y_train)
    rfe.support_
    list(zip(X_train.columns, rfe.support_, rfe.ranking_))
    col = X_train.columns[rfe.support_]
    X_train.columns[~rfe.support_]
    X_train_sm = sm.add_constant(X_train[col])
    logm2 = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())
    res = logm2.fit()

    # Getting the predicted values on the train set
    y_train_pred = res.predict(X_train_sm)
    y_train_pred = y_train_pred.values.reshape(-1)

    y_train_pred_final = pd.DataFrame({'Survived': y_train.values, 'Survived_prob': y_train_pred})
    y_train_pred_final['PassengerId'] = y_train.index
    y_train_pred_final.head()

    y_train_pred_final['predicted'] = y_train_pred_final.Survived_prob.map(lambda x: 1 if x > 0.54 else 0)

    # Let's see the head
    # Let's check the overall accuracy.
    print(str(metrics.accuracy_score(y_train_pred_final.Survived, y_train_pred_final.predicted)))
    X_test = X_test[col]
    X_test.head()
    X_test_sm = sm.add_constant(X_test)
    y_test_pred = res.predict(X_test_sm)
    y_test_pred[:10]
    ## Converting y_pred to a dataframe which is an array
    y_pred_1 = pd.DataFrame(y_test_pred)
    y_pred_1.head()
    # Removing index for both dataframes to append them side by side

    y_pred_1.reset_index(drop=True, inplace=True)
    # creating a final
    y_pred_final = pd.concat([y_pred_1], axis=1)
    y_pred_final.head()
    y_pred_final = y_pred_final.rename(columns={0: 'Survived_Prob'})
    y_pred_final.head()
    y_pred_final['Survived'] = y_pred_final.Survived_Prob.map(lambda x: 1 if x > 0.54 else 0)

    return toText(y_pred_final.head().to_string())


@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')


if __name__ == '__main__':
    app.run(debug=False)
