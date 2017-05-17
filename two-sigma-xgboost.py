import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xbs
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from sklearn import preprocessing
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import winsound

train = pd.read_json('train.json')
test = pd.read_json('test.json')
df_all = pd.concat([train, test], keys=['df_train', 'df_test'])
df_all.drop(['street_address', 'display_address'],axis=1,inplace=True)#features not used

#modifications which can be made on train and test data at once
#number of photos
df_all['photos']=df_all['photos'].apply(len)

#bumber of features
df_all['features']=df_all['features'].apply(len)

#length of description
df_all['description_len'] = df_all['description'].str.len()
df_all['description_len'].fillna(0)
df_all.drop(['description'],axis=1 , inplace =True)

'''manager, bulding id - counts will be used instead of labels
le = preprocessing.LabelEncoder()
df_all['manager_id'] = le.fit_transform(df_all['manager_id'])
df_all['building_id'] = le.fit_transform(df_all['building_id'])'''

# price per bathroom and bedroom
df_all['price_per_bathroom'] = df_all['price']/df_all['bathrooms']
df_all['price_per_bedroom'] = df_all['price']/df_all['bedrooms']

#split data back to test and train part
train = df_all.loc['df_train'].copy()
test = df_all.loc['df_test'].copy()
#extract result vector and map values to numbers
y = train['interest_level']
train.drop(['interest_level'],axis=1 , inplace =True)
test.drop(['interest_level'],axis=1 , inplace =True)
y =y.map( {'low': 0, 'medium': 1, 'high': 2} ).astype(int)

#modifications which should be made on train and test separately

#age of listing in days
train["created"]  = pd.to_datetime(train["created"])
min_created_train = min(train["created"])
train["created"] = train["created"].apply(lambda x: x-min_created_train)
train["created"] = train["created"].apply(lambda x: x.days)
#print(max(train["created"]) #88
test["created"]  = pd.to_datetime(test["created"])
min_created_train = min(test["created"])
test["created"] = test["created"].apply(lambda x: x-min_created_train)
test["created"] = test["created"].apply(lambda x: x.days)
#print(max(test["created"]) #88

#numbers of listings belongig to manager
def getCounts(df, columnName):
    counts = pd.Series([0] * df.size, index=df)
    c = df.value_counts()
    for i in c.keys():
        counts.loc[i] = c[i]
    res_pd = pd.DataFrame(counts.values, index=df.index, columns=[columnName])
    return  pd.concat([df, res_pd], axis=1)

train = pd.concat([train, getCounts(train['manager_id'], 'listing_of_manager_couts')], axis=1)
#print(max(train['listing_of_manager_couts'])) #2533
test = pd.concat([test, getCounts(test['manager_id'], 'listing_of_manager_couts')], axis=1)
#print(max(test['listing_of_manager_couts'])) #3854
#numbers of relating to building id
train = pd.concat([train, getCounts(train['building_id'], 'listing_per_building')], axis=1)
#print(max(train['listing_per_building'])) #8286
test = pd.concat([test, getCounts(test['building_id'], 'listing_per_building')], axis=1)
#print(max(train['listing_per_building'])) #8286
test.drop(['manager_id', 'building_id'],axis=1,inplace=True)
train.drop(['manager_id', 'building_id'],axis=1,inplace=True)

#read new features obtained from classification of texts (see files gridsearch-descriptiom and predict-from-features.py)
train['pred_features'] = pd.read_csv('pred_features', index_col = 0)
train['pred_description'] = pd.read_csv('pred_description', index_col = 0)
test['pred_features'] = pd.read_csv('pred_feature_test', index_col = 0)
test['pred_description'] = pd.read_csv('pred_description_test', index_col = 0)

'''
#grid search
model= xbs.XGBClassifier(max_delta_step=1, objective='multi:softprob', learning_rate = 0.01)
params = { 'n_estimators': [500,  1000, 3000],
            'reg_lambda': [0, 10, 100],
            'max_depth': [3 , 6, 8],
            'subsample':[0.7 , 0.8,  0.9],
            'colsample_bytree': [0.7 , 0.8,  0.9]
                }
grid_search = GridSearchCV(model, params, verbose=1, scoring = 'neg_log_loss',cv = 5)

def fill_res(x):
    if np.isnan(x) :
        return 1
    else:
        return 0

best_params = {'n_estimators': [],
            'reg_lambda': [],
            'max_depth': [],
            'subsample': [],
            'colsample_bytree': []
              }
n=10
for rs in range(0, n):
    X_train, X_test, y_train, y_test = train_test_split(train , y , test_size=0.2, random_state=rs)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.predict(X_test)
    yproba = grid_search.predict_proba(X_test)
    print("log loss")
    print(log_loss(y_test, yproba))
    winsound.Beep(2500, 1000)
    cc = (y_pred - y_test) / (y_pred - y_test)
    cc = cc.apply(lambda x: fill_res(x))
    print("accuracy")
    print(cc.mean())
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print(grid_search.best_params_ )
    for key in best_params:
        best_params[key].append(grid_search.best_params_[key])

for key in best_params:
    print("mean" + key + ": {0}".format(np.mean(best_params['n_estimators'])))
'''

#predict interest levels and prepare submission
clf = xbs.XGBClassifier(max_delta_step=1, objective='multi:softprob', learning_rate = 0.01,n_estimators=3000, reg_lambda = 0, max_depth = 8, subsample = 0.7, colsample_bytree = 0.7)
clf = clf.fit(train, y )
y_pred = clf.predict(test)
yproba = clf.predict_proba(test)
test2 = pd.read_json('test2.json')
results = pd.DataFrame({'listing_id': test2['listing_id'], 'high': yproba[:, 2], 'medium': yproba[:, 1], 'low': yproba[:, 0]})
results = results[['listing_id', 'high', 'medium', 'low']]
eps = 10**(-15)
results['high'] = results['high'].apply(lambda x: max(min(x,1-eps),eps))
results['medium'] = results['medium'].apply(lambda x: max(min(x,1-eps),eps))
results['low'] = results['low'].apply(lambda x: max(min(x,1-eps),eps))
results.to_csv('model_xgb', index=False)
print("done")
winsound.Beep(2500, 1000)



