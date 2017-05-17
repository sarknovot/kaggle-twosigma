import pandas as pd
import numpy as np
import winsound
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import normalize
from split_data import split_data
from sklearn.metrics import confusion_matrix
from sklearn import svm

def process_features(df):
    df = df.apply(lambda x: [i.replace(" ", "") for i in x])
    df = df.apply(lambda x: ' '.join(x))
    #df = df.apply(lambda x: x.lower()) countVectorizer converts characters to lowercase
    vectorizer = CountVectorizer()
    train_features  = vectorizer.fit_transform(df)
    words_sum = train_features.sum(axis=0)
    words_counts = np.squeeze(np.asarray(words_sum))
    words_counts_series = pd.Series(words_counts, index=vectorizer.get_feature_names())
    words_counts_series.sort_values(inplace=True, ascending=False)
    words_counts_series = pd.Series(words_counts_series.index, index=words_counts_series.values)
    return vectorizer, train_features, words_counts_series

#read train data
train = pd.read_json('train.json')
description = train['description'].copy()
#get result vector from training data
result = train['interest_level'].copy()
result =result.map( {'low': -1, 'medium': 0 , 'high': 1} ).astype(int)

vectorizer, train_features, words_counts = process_features(train['features'])
number_of_words = 200
ch2 = SelectKBest(chi2, number_of_words)
best_words = ch2.fit_transform( train_features , result)
best_words = ch2.get_support(indices=True)

#use only words that are common to both training and test set
test = pd.read_json('test.json')
names = vectorizer.get_feature_names()
names_best = [names[i] for i in best_words]
test_vectorizer, test_features, test_words_counts = process_features(test['features'])
test_names = test_vectorizer.get_feature_names()
mask = [n in test_names for n in names_best]
mask = np.array(mask)
best_words = np.array(best_words)
best_words = best_words[mask]
common_words = [names[i] for i in best_words]

#get train features
train_features = train_features[:,best_words]
train_features = normalize(train_features, norm='l2', axis=0)

#get test values
test['description'] = test['description'].apply(lambda x: [i.replace(" ", "") for i in x])
test['description'] = test['description'].apply(lambda x: ' '.join(x))
vectorizer = CountVectorizer( vocabulary = common_words)
test_features = vectorizer.fit_transform(test['description'])
test_features = normalize(test_features, norm='l2', axis=0)
'''
#Split data so that 1/5 is used for validation, then perform 5 grid searches to find best parameters of SVC,
#the same splitting will be then used for prediction of result based on description parameter
splits_no = 5
x_train, y_train, x_test, y_test = split_data(splits_no, train_features, result)


#Find best parameters
#Define space of parameters which will be went through
clf = svm.SVC(decision_function_shape='ovr', class_weight = 'balanced', kernel = 'linear')
grid_search = GridSearchCV(clf, {
                    'C': [   1 , 100 ] ,
                    'gamma': [0.05 , 0.3 ],
                }, verbose=1, scoring = 'f1_macro')
#Find best parameters for every part of data - should be performed several times, however, solution was quite stable
#save results of predictions to be later used as a featu
features_new_feature = pd.Series()
for i in range(0, splits_no):
    grid_search.fit(x_train[i],y_train[i])
    y_pred = grid_search.predict(x_test[i])
    print(confusion_matrix(y_test[i], y_pred ))
    features_new_feature = features_new_feature.append(pd.Series(y_pred, index=y_test[i].index))
#save results of predictions to be later used as a feature of train data - proper index has to be set
train['pred_feature'] = features_new_feature
train['pred_feature'] =train['pred_feature'].map( {-1: 0, 0: 1, 1: 2} ).astype(int)
pred_description = pd.Series(train['pred_feature'].values, train.index)
#pred_description.to_csv('pred_feature_train')
winsound.Beep(2500, 1000)
'''

#create new feature for test data
clf = svm.SVC(decision_function_shape='ovr', class_weight = 'balanced', kernel = 'linear', C=100, gamma = 0.03)
clf = clf.fit(train_features, result)
y_pred = clf.predict(test_features)
#save results of predictions to be later used as a feature of train data - proper index has to be set
test['pred_feature'] = y_pred
test['pred_feature'] =test['pred_feature'].map( {-1: 0, 0: 1, 1: 2} ).astype(int)
pred_description = pd.Series(test['pred_feature'].values, test.index)
pred_description.to_csv('pred_feature_test')
winsound.Beep(2500, 1000)