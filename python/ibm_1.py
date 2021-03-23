from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import scikitplot as skplt

import numpy
import pandas
import sklearn
from sklearn import impute
from sklearn import preprocessing
from sklearn import pipeline
from sklearn import compose

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

area = 75
x = df_churn['ESTINCOME']
y = df_churn['DAYSSINCELASTTRADE']
z = df_churn['TOTALDOLLARVALUETRADED']

pop_a = mpatches.Patch(color='#BB6B5A', label='High')
pop_b = mpatches.Patch(color='#E5E88B', label='Medium')
pop_c = mpatches.Patch(color='#8CCB9B', label='Low')

"""
convert label # to hexcode color
"""
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

fig = plot.figure(figsize = (12, 6))
fig.suptitle('2D and 3D view of churn risk data')

# first sub plot
ax_2D = fig.add_subplot(1, 2, 1)

ax_2D.scatter(x, y, alpha = 0.8, c = colormap(label), s = area)

ax_2D.set_ylabel('DAYS SINCE LAST TRADE')
ax_2D.set_xlabel('ESTIMATED INCOME')

plot.legend(handles = [pop_a, pop_b, pop_c])

# second sub plot
ax_3D = fig.add_subplot(1, 2, 2, projection = '3d')

ax_3D.scatter(z, x, y, c = colormap(label), marker = 'o')

ax_3D.set_xlabel('TOTAL DOLLAR VALUE TRADED')
ax_3D.set_ylabel('ESTIMATED INCOME')
ax_3D.set_zlabel('DAYS SINCE LAST TRADE')

plot.legend(handles = [pop_a, pop_b, pop_c])

fig.savefig('fancy_plot.pdf',  bbox_inches='tight')