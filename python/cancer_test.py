import numpy
import pandas
import sklearn
import matplotlib.pyplot as plot
import seaborn

from sklearn import datasets
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn import inspection

# load data set
cancer = datasets.load_breast_cancer()

df = pandas.DataFrame(cancer.data, columns = cancer.feature_names)
df['class'] = pandas.Categorical.from_codes(cancer.target, cancer.target_names)

# show class distribution
print('Number of each target class is ')
print(df.groupby('class').size())

# output features
print(df.head())

# plot distrbutions
# seaborn.pairplot(data = df, hue = 'class').savefig('plots/cancer/pairs.pdf')

# create feature and target data frames
features = df.drop(['class'], axis = 1)
target = pandas.Series(cancer.target)

# split into train and test sets
splits = model_selection.train_test_split(features, target, random_state = 0)
X_train, X_test, y_train, y_test = splits

# define model
lgr = linear_model.LogisticRegression(
    random_state = 0,
    penalty = 'l2',
    C = 1,
    tol = 1e-4,
    max_iter = 100,
    n_jobs = -1,
)

lrc = pipeline.Pipeline(
    steps = [
        ('scale', preprocessing.StandardScaler()),
        ('class', lgr),
    ]
)

rfc = ensemble.RandomForestClassifier(
    random_state = 0,
    max_features = 'sqrt',
    n_estimators = 100,
    n_jobs = -1,
)

classifiers = {
    'LogisticReg' : lrc,
    # 'RandomForest' : rfc,
    }

for name, classifier in classifiers.items():
    # do cross validation
    scores = model_selection.cross_val_score(
        classifier,
        X_train,
        y_train,
        cv = 10,
    )
    print(name + ' has %0.2f ± %0.2f accuracy' % (scores.mean(), scores.std()))

    # actually train
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # look at metrics
    report = metrics.classification_report(y_test, y_pred)
    print(report)

    # plot confusion matrix
    cm = metrics.plot_confusion_matrix(
        classifier,
        X_test,
        y_test,
        normalize = 'pred',
        cmap = 'Greens',
    )
    path = 'plots/cancer/cm_' + name + '.pdf'
    cm.figure_.savefig(path, bbox_inches = 'tight')

    # partial dependence plots
    for i in [0, 1]:
        pdp = inspection.plot_partial_dependence(
            classifier,
            X_train,
            features = list(range(0, 30)),
            target = i,
            n_jobs = -1,
        )
        path = 'plots/cancer/pdp_' + str(i) + '_' + name + '.pdf'
        pdp.figure_.savefig(path, bbox_inches = 'tight')
    # end
# end