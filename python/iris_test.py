import numpy
import pandas
import sklearn
import matplotlib.pyplot as plot

from sklearn import datasets
from sklearn import model_selection
from sklearn import pipeline
from sklearn import ensemble
from sklearn import metrics


# load iris into data frame
iris = datasets.load_iris()

df = pandas.DataFrame(iris.data, columns = iris.feature_names)
df['target'] = pandas.Series(iris.target)
df['species'] = pandas.Categorical.from_codes(iris.target, iris.target_names)

# show class distribution
print("Number of each target class is ")
print(df.groupby('target').size())

# output features
print(df.head())

# plot distributions
# plot X[i] vs y
# plot X[i] vs X[j]

def plot_scatter(x, y, df):
    fig = plot.figure()

    plot.scatter(x, y, data = df, c = 'target')

    plot.xlabel(x)
    plot.ylabel(y)

    fig.savefig('plots/' + x + '_vs_' + y + '.pdf',  bbox_inches='tight')

for i in range(0, len(iris.feature_names)):
    x = iris.feature_names[i]
    # plot_scatter(x, 'target')

    for j in range(i+1, len(iris.feature_names)):
        y = iris.feature_names[j]
        plot_scatter(x, y, df)
    # end
# end

# create feature and target dataframes
features = df.drop(['target', 'species'], axis = 1)
target = df['target']

# for regression 
# do pipeline with PCA
# and be able to show it satisfies the four requirement

# talk about missing values
# normalization
# onehot encoding

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, random_state = 0)

rfc = ensemble.RandomForestClassifier(
    n_estimators = 100,
    random_state = 0,
    n_jobs = -1,
)

# mention pipeline
# do cross validation

rfc.fit(X_train, y_train)
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

