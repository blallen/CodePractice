import numpy
import pandas
import sklearn
import seaborn
import matplotlib.pyplot as plot

from sklearn import datasets
from sklearn import model_selection
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
from sklearn import inspection

# load iris into data frame
iris = datasets.load_iris()

df = pandas.DataFrame(iris.data, columns = iris.feature_names)
df['species'] = pandas.Categorical.from_codes(iris.target, iris.target_names)

# show class distribution
print("Number of each target class is ")
print(df.groupby('species').size())

# output features
print(df.head())

# plot distributions
seaborn.pairplot(data=df, hue='species').savefig('plots/iris_pairs.pdf')

# create feature and target dataframes
features = df.drop(['species'], axis = 1)
target = pandas.Series(iris.target)

# for regression 
# do pipeline with PCA
# and be able to show it satisfies the four requirement

# talk about missing values
# normalization
# onehot encoding

# split into train and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, random_state = 0)

# define models
lgr = linear_model.LogisticRegression(
    random_state = 0,
    penalty = 'l2',
    C = 1,
    tol = 1e-4,
    max_iter = 100,
    n_jobs = -1,
)

# mention pipeline
lrc = pipeline.Pipeline(
    steps = [
        ('scale', preprocessing.StandardScaler()),
        ('class', lgr)
    ],
)

rfc = ensemble.RandomForestClassifier(
    random_state = 0,
    max_features = 'sqrt',
    n_estimators = 100, 
    n_jobs = -1,
)

gbc = ensemble.GradientBoostingClassifier(
    random_state = 0,
    max_features = 'sqrt',
    n_estimators = 100,
    learning_rate = 0.1, 
    max_depth = 3,
)

classifiers = {
    'LogisticReg' : lrc,
    'RandomForest' : rfc, 
    'GradientBoost' : gbc,
}
for name, classifier in classifiers.items():
    # do cross validation
    scores = model_selection.cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=10,
    )
    print(name + ' model has %0.2f accuracy with a standard deviation of %0.2f' % (scores.mean(), scores.std()))

    # actually train
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # look at metrics
    report = metrics.classification_report(y_test, y_pred)
    print('The classification report for ' + name + ' model:')
    print(report)

    # plot confusion matrix
    cm = metrics.plot_confusion_matrix(
        classifier, 
        X_test,
        y_test,
        # normalize = 'pred',
        cmap = 'Greens',
    )
    path = 'plots/cm_' + name + '.pdf'
    cm.figure_.savefig(path,  bbox_inches='tight')

    # plot partial dependence plots
    for i in [0, 1, 2]:
        pdp1D = inspection.plot_partial_dependence(
            classifier,
            X_train,
            features = [0, 1, 2, 3],
            target = i,
            n_jobs = -1,
        )
        path = 'plots/pdp1D_' + str(i) + '_' + name + '.pdf'
        pdp1D.figure_.savefig(path,  bbox_inches='tight')

        """
        to_plot = []
        for j in range(0,4):
            for k in range(j+1, 4):
                to_plot.append((j,k))

        pdp2D = inspection.plot_partial_dependence(
            classifier,
            X_train,
            features = to_plot,
            target = i,
            n_jobs = -1,
        )
        path = 'plots/pdp2D_' + str(i) + '_' + name + '.pdf'
        pdp2D.figure_.savefig(path,  bbox_inches='tight')
        """
    # end
# end

