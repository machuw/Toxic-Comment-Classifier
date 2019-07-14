import sys, os, re, csv, codecs
import re, string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics

# load data
train = pd.read_csv('data/train.csv')
test_comments = pd.read_csv('data/test.csv')
test_labels = pd.read_csv('data/test_labels.csv')
subm_sample = pd.read_csv('data/sample_submission.csv')

#test = pd.concat([test_comments, test_labels], axis=1)
#test = test[test.toxic != -1]
test = test_comments 
 
# split data 
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['clean'] = 1 - train[class_names].max(axis=1)

# fillna
train['comment_text'].fillna('unknown', inplace=True)
test['comment_text'].fillna('unknown', inplace=True)

# create bag of word using ngrams
word_vec = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 2),
    max_features=10000,
    min_df=3, max_df=0.9,
    use_idf=True,
    smooth_idf=True
)
#all_comments = pd.concat([train['comment_text'], test['comment_text']])
#word_vec.fit(all_comments)

train_word_features = word_vec.fit_transform(train['comment_text'])
test_word_features = word_vec.transform(test['comment_text'])

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = train_word_features
test_x = test_word_features

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

preds = np.zeros((len(test), len(class_names)))

for i, j in enumerate(class_names):
    print('fit ', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]

test_y = test_labels.iloc[:, 1:]

def score_f(y, p_y):
    print("roc auc score ", metrics.roc_auc_score(y, p_y))
    print("accuracy ", metrics.accuracy_score(y, p_y))
    print("precision score ", metrics.precision_score(y, p_y, average='weighted'))
    print("recall score ", metrics.recall_score(y, p_y, average='weighted'))
    print("f1 score ", metrics.f1_score(y, p_y, average='weighted'))

for i, j in enumerate(class_names):
    print("\n", j)
    #score_f(test_y.iloc[:,i], preds[:,i])    
    #print(metrics.classification_report(test_y.iloc[:,i], preds[:,i]))

#print("\nOverall")
#score_f(test_y, preds)
#print(metrics.classification_report(test_y, preds))

submid = pd.DataFrame({'id': subm_sample["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns=class_names)], axis=1)
submission.to_csv('submission.csv', index=False)