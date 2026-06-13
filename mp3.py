# Starter code for DSC 240 MP3
import random

import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

import sklearn

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

np.random.seed(0)

def compute_metric(labels, expected):
    labels = np.array(labels)
    expected = np.array(expected)
    
    tp = np.sum(labels[expected == 1])
    fp = np.sum(labels[expected == 0])
    tn = np.sum(1-labels[expected == 0])
    fn = np.sum(1-labels[expected == 1])
    
    tpr = tp/(tp+fn) if (tp+fn) > 0 else 0
    fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
    error_rate = (fp+fn)/(tp+fp+tn+fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0

    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "error_rate": error_rate,
    }


def run_train_test(training_data: pd.DataFrame, testing_data: pd.DataFrame) -> List[int]:
    
    #preprocess features
    X_train = training_data.drop('target', axis=1)
    y_train = training_data['target']
    X_test = testing_data

    #define which columns are which
    numeric_features = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
    categorical_features = ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                            'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'QUANTIZED_INC', 
                            'QUANTIZED_AGE', 'QUANTIZED_WORK_YEAR']
    binary_features = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_MOBIL', 
                       'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']

    #build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', 'passthrough', binary_features)])

    #create the classifier and define 'clf'
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=0))])

    #update the parameters (Train)
    clf.fit(X_train, y_train)

    #evaluate/predict
    predict = np.ones(len(testing_data))
    predict = clf.predict(X_test)

    return predict.astype(int)


if __name__ == '__main__':
 
    training = pd.read_csv('./train.csv')
    development = pd.read_csv('./dev.csv')

    target_label = development['target']
    
    development.drop('target', axis=1, inplace=True)
    
    prediction = run_train_test(training, development)
    
    target_label = target_label.values
    status = compute_metric(prediction, target_label)
    print(status)



    


