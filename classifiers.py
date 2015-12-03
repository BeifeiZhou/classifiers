from sklearn import svm
import pandas as pd
import numpy as np
import random
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv('../input/train.csv')
factors = list(set(data['Product_Info_2'].tolist()))
factors2num = dict(zip(factors, range(len(factors))))
data['Product_Info_2'] = map(lambda x: factors2num[x], data['Product_Info_2'])
data = data.fillna(0)

###generate data
def splitx(n1, n2, x):
    nrows = x.shape[0]
    nslice = int(n1/float(n1+n2)*nrows)
    list1 = random.sample(range(nrows), nslice)
    x1 = x.loc[list1,:]
    list2 = list(set(range(nrows))-set(list1))
    x2 = x.loc[list2,:]
    return (x1, x2)

(train, test) = splitx(6, 4, data)
clf = svm.SVC()
X = train[range(126)]
Y = train[[127]]
clf.fit(X,Y)
X_test = test[range(126)]
Y_test = test['Response']
Y_pred = clf.predict(X_test)

a = np.array(Y_test.tolist())-Y_pred
print 'SVM: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))

clf = SGDClassifier(loss="hinge", penalty="l2")
Y_pred = clf.predict(X_test)
a = np.array(Y_test.tolist())-Y_pred
print 'SGD: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
Y_pred = clf.predict(X_test)
a = np.array(Y_test.tolist())-Y_pred
print 'tree: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
Y_pred = clf.predict(X_test)
a = np.array(Y_test.tolist())-Y_pred
print 'RF: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
clf = clf.fit(X, Y)
Y_pred = clf.predict(X_test)
a = np.array(Y_test.tolist())-Y_pred
print 'ExtraTree: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))

clf = AdaBoostClassifier(n_estimators=100)
clf = clf.fit(X, Y)
Y_pred = clf.predict(X_test)
a = np.array(Y_test.tolist())-Y_pred
print 'AdaBoost: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf = clf.fit(X, Y)
Y_pred = clf.predict(X_test)
a = np.array(Y_test.tolist())-Y_pred
print 'GradientBoost: the precision is: '+str((len(a)-np.count_nonzero(a))/float(len(a)))
