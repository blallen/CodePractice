import numpy
import pandas
import sklearn
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches

from sklearn import impute
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import compose
from sklearn import model_selection
from sklearn import metrics
from sklearn import ensemble
from sklearn import datasets

iris = datasets.load_iris()

print(type(iris)) # can convert to data frame

X = iris['data']
y = iris['target']

print(X.shape)
print(y.shape)

# for regression 
# do pipeline with PCA
# and be able to show it satisfies the four requirement

# show class distribution
# output features

# plot X[i] vs y
# plot X[i] vs X[j]

# df = pandas.DataFrame(data = X, columns = ['one', 'two', 'three', 'four'])
# df['target'] = y

# print(df.head())

# talk about missing values
# normalization
# onehot encoding

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 0)

rfc = ensemble.RandomForestClassifier(
    n_estimators = 100,
    random_state = 0,
    n_jobs = -1,
)

# mention pipeline

rfc.fit(X_train, y_train)

# do cross validation

y_pred = rfc.predict(X_test)

report = metrics.classification_report(y_test, y_pred)
print("The classification report for the model : \n\n")
print(report)

disp = metrics.plot_confusion_matrix(
    rfc, 
    X_test,
    y_test,
    # normalize = 'pred',
    cmap = 'Greens',
)
disp.figure_.savefig('confusion_matrix.pdf',  bbox_inches='tight')

# grad boost
# adaboost

