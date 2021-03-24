from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

import numpy
import pandas
import sklearn
import matplotlib.pyplot as plot
import matplotlib.patches as mpatches
import scikitplot as skplot

from sklearn import impute
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import compose
from sklearn import model_selection
from sklearn import metrics

###############
## Get the data
###############

# import data
df_churn = pandas.read_csv('mergedcustomers_missing_values_GENDER.csv')

#remove columns that are not required
df_churn = df_churn.drop(['ID'], axis=1)

print(df_churn.head())

"""
###############
## Basic Counting
###############

print("The dataset contains columns of the following data types : \n" +str(df_churn.dtypes))

print("The dataset contains following number of records for each of the columns : \n" +str(df_churn.count()))


print( "Each category within the churnrisk column has the following count : ")
print(df_churn.groupby(['CHURNRISK']).size())
# .count() will do a count in each column, lots of redudant data
# .size() will do the same count, but only display once for each value of 'CHURNRISK'

###############
## Basic Plotting
###############

#bar chart to show split of data
index = ['High','Medium','Low']
color = ['#BB6B5A','#8CCB9B','#E5E88B']
title = "Total number for occurences of churn risk " + str(df_churn['CHURNRISK'].count())

# fig size is in inches x.x
churn_plot = df_churn['CHURNRISK'] \
                .value_counts(sort = True, ascending = False) \
                .plot(kind = 'bar', figsize=(4,4), title = title, color = color)
churn_plot.set_xlabel("Churn Risk")
churn_plot.set_ylabel("Frequency")

churn_plot.get_figure().savefig('churn_plot.pdf',  bbox_inches='tight')
# bbox_inches='tight' is magic
"""

###############
## Data Cleaning
###############

# Defining the categorical columns 
columns_categorical = ['GENDER', 'STATUS', 'HOMEOWNER']

# print("Categorical columns : ")
# print(columns_categorical)

impute_categorical = impute.SimpleImputer(strategy = 'most_frequent')
# why not mode? who knows?

onehot_categorical = preprocessing.OneHotEncoder(
    handle_unknown = 'error',
    drop = 'if_binary',
)

transformer_categorical = pipeline.Pipeline(
    steps = [
        ('impute', impute_categorical), 
        ('onehot', onehot_categorical),
    ]
)

# Defining the numerical columns 
columns_numerical = df_churn.select_dtypes(
    include = [numpy.float, numpy.int]
).columns

# print("Numerical columns : ")
# print(columns_numerical)

scaler_numerical = preprocessing.StandardScaler()

transformer_numerical = pipeline.Pipeline(steps = [('scale', scaler_numerical)])

# start setting up preprocessors

preprocessor_categorical = compose.ColumnTransformer(
    transformers = [('cat', transformer_categorical, columns_categorical)],
    remainder = 'passthrough',
)

preprocessor_numerical = compose.ColumnTransformer(
    transformers = [('num', transformer_numerical, columns_numerical)],
    remainder = 'passthrough',
)

preprocessor_all = compose.ColumnTransformer(
    transformers = [
        ('cat', transformer_categorical, columns_categorical),
        ('num', transformer_numerical, columns_numerical),
    ],
    remainder = 'passthrough'
)

"""
# The transformation happens in the pipeline. Temporarily done here to show what intermediate value looks like.

# ColumnTransformer.fit_transform() returns a numpy.ndarray
# wrap with pandas.DataFrame for more utility
df_churn_cat = pandas.DataFrame(preprocessor_categorical.fit_transform(df_churn))
print("Data after transforming categorical columns:")
print(df_churn_cat.head())

df_churn_num = pandas.DataFrame(preprocessor_numerical.fit_transform(df_churn))
print("Data after transforming numerical columns:")
print(df_churn_num.head())

df_churn_all = pandas.DataFrame(preprocessor_all.fit_transform(df_churn))
print("Data after transforming all columns:")
print(df_churn_all.head())
"""

# prepare data frame for splitting into train and test sets

features = df_churn.drop(['CHURNRISK'], axis = 1)

label_churn = pandas.DataFrame(df_churn, columns = ['CHURNRISK'])
label_encoder = preprocessing.LabelEncoder()
label = df_churn['CHURNRISK']

label = label_encoder.fit_transform(label)
# print("Encoded value of Churnrisk after applying label encoder : " + str(label))

###############
## Fancy Plotting
###############

'''
convert label # to hexcode color
'''
def colormap(risk_list):
    cols=[]

    for l in risk_list:
        if l == 0: # high
            cols.append('#BB6B5A')
        elif l == 2: # medium 
            cols.append('#E5E88B')
        elif l == 1: # low
            cols.append('#8CCB9B')
            
    return cols

pop_a = mpatches.Patch(color='#BB6B5A', label='High')
pop_b = mpatches.Patch(color='#E5E88B', label='Medium')
pop_c = mpatches.Patch(color='#8CCB9B', label='Low')

handles = [pop_a, pop_b, pop_c]

"""
area = 75
x = df_churn['ESTINCOME']
y = df_churn['DAYSSINCELASTTRADE']
z = df_churn['TOTALDOLLARVALUETRADED']

fig = plot.figure(figsize = (12, 6))
fig.suptitle('2D and 3D view of churn risk data')

# first sub plot
ax_2D = fig.add_subplot(1, 2, 1)

ax_2D.scatter(x, y, alpha = 0.8, c = colormap(label), s = area)

ax_2D.set_ylabel('DAYS SINCE LAST TRADE')
ax_2D.set_xlabel('ESTIMATED INCOME')

plot.legend(handles = handles)

# second sub plot
ax_3D = fig.add_subplot(1, 2, 2, projection = '3d')

ax_3D.scatter(z, x, y, c = colormap(label), marker = 'o')

ax_3D.set_xlabel('TOTAL DOLLAR VALUE TRADED')
ax_3D.set_ylabel('ESTIMATED INCOME')
ax_3D.set_zlabel('DAYS SINCE LAST TRADE')

plot.legend(handles = handles)

fig.savefig('fancy_plot.pdf',  bbox_inches='tight')
"""

###############
## Split data
###############

X_train, X_test, y_train, y_test = model_selection.train_test_split(features, label, random_state = 0)

print("Dimensions of datasets that will be used for training : Input features" + str(X_train.shape) + " Output label" + str(y_train.shape))
print("Dimensions of datasets that will be used for testing : Input features" + str(X_test.shape) + " Output label" + str(y_test.shape))

def compared_2D(X_test, y_test, y_pred, model_name, handles):
    fig = plot.subplots(ncols = 2, figsize = (10, 4))

    score = metrics.accuracy_score(y_test, y_pred)
    suptitle = 'Actual vs Predicted data : ' + model_name + '. Accuracy : %.2f' % score
    fig.suptitle(suptitle)

    ax_test = fig.add_subplot(121)
    ax_test.scatter(
        X_test['ESTINCOME'],
        X_test['DAYSSINCELASTTRDE'], 
        alpha = 0.8,
        c = colormap(y_test),
    )

    ax_test.set_xlabel('ESTIMATED INCOME')
    ax_test.set_ylabel('DAYS SINCE LAST TRADE')

    plot.title('Actual')
    plot.legend(handles = handles)

    ax_pred.subplot(122)
    ax_pred.scatter(
        X_test['ESTINCOME'],
        X_test['DAYSSINCELASTTRDE'], 
        alpha = 0.8,
        c = colormap(y_pred),
    )

    ax_pred.set_xlabel('ESTIMATED INCOME')
    ax_pred.set_ylabel('DAYS SINCE LAST TRADE')

    plot.title('Predicted')
    plot.legend(handles = [pop_a, pop_b, pop_c])

    fig.savefig(model_name + '_2D.pdf',  bbox_inches='tight')

def compare_3D(X_test, y_test, y_pred, model_name, handles):
    fig = plot.figure(figsize = (12, 10))

    score = metrics.accuracy_score(y_test, y_pred)
    suptitle = 'Actual vs Predicted data : ' + model_name + '. Accuracy : %.2f' % score
    fig.suptitle(suptitle)

    ax_test = fig.add_subplot(121, projection = '3d')
    ax_test.scatter(
        X_test['TOTALDOLLARVALUETRADED'],
        X_test['ESTINCOME'],
        X_test['DAYSSINCELASTTRDE'],
        alpha = 0.8,
        c = colormap(y_test),
        marker = 'o',
    )

    ax_test.set_xlabel('TOTAL DOLLAR VALUE TRADED')
    ax_test.set_ylabel('ESTIMATED INCOME')
    ax_test.set_zlabel('DAYS SINCE LAST TRADE')

    plot.legend(handles = [pop_a, pop_b, pop_c])
    plot.title('Actual')

    ax_pred = fig.add_subplot(122, projection = '3d')
    ax_pred.scatter(
        X_test['TOTALDOLLARVALUETRADED'],
        X_test['ESTINCOME'],
        X_test['DAYSSINCELASTTRDE'],
        alpha = 0.8,
        c = colormap(y_pred),
        marker = 'o',
    )

    ax_pred.set_xlabel('TOTAL DOLLAR VALUE TRADED')
    ax_pred.set_ylabel('ESTIMATED INCOME')
    ax_pred.set_zlabel('DAYS SINCE LAST TRADE')

    plot.legend(handles = [pop_a, pop_b, pop_c])
    plot.title('Predicted')

    fig.savefig(model_name + '_3D.pdf',  bbox_inches='tight')

def model_metrics(y_test, y_pred):
    print("Decoded values of churn risk after applying inverse of label encoder : " + str(numpy.unique(y_pred)))

    fig = skplot.metrics.plot_confusion_matrix(
        y_test,
        y_pred,
        text_fontsize = 'small',
        cmap = 'Greens',
        figsize = (6, 4)
    )
    fig.savefig('model_metrics.pdf',  bbox_inches='tight')

    report = metrics.classification_report(y_test, y_pred)
    print("The classification report for the model : \n\n")
    print(report)

###############
## Build model
###############