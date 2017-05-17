'''
By splitting train_features matrix and result vector the function split_data crates splits_no new data sets.
New data sets consist of data having size ((splits_no-1)/splits_no*sizeof(train_features)),
which can be used for training, and rest of data (of size 1/splits_no*sizeof(train_features)),
which can be used for validation.
'''
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
import pandas as pd

'''
splits_no - number of data splits
train_features - data matrix which should be split
result - result vector corresponding to train_features data
'''
def split_data(splits_no, train_features, result):
    #Sets parameters of splitting
    test_part = 1/splits_no
    train_part = 1 - test_part

    #Split data
    x_test = []
    y_test  = []
    for i in range(0,splits_no-1):
        test_size =  test_part/(1-i*test_part)
        #print(test_size)
        trainx, testx, trainy, testy = train_test_split(train_features, result, random_state=i, test_size=test_size)
        x_test .append(testx)
        y_test .append(testy)
        train_features = trainx
        result = trainy
    x_test .append(trainx)
    y_test .append(trainy)

    #Group train data
    x_train= []
    y_train= []
    for i in range(0,splits_no):
        x_train.append(vstack(x_test[0:i]+x_test[i+1:splits_no]))
        y_train.append(pd.concat(y_test[0:i]+y_test[i+1:splits_no], axis=0))
    return x_train, y_train, x_test, y_test