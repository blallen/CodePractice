import pandas as pd
import numpy as np
from matplotlib import pyplot

##############
# Exercise 1 #
############## 

## load data
input_data = pd.read_csv("screening_exercise_orders_v202101.csv", parse_dates = ['date'])

# check labels and such
print(
"""
##############
# Exercise 0 #
##############
"""
)
print(input_data.head(5))

### create desired output data frame
output_data = \
    input_data.groupby('customer_id') \
        .agg(
            gender=('gender', 'mean'),
            most_recent_order_date=('date', 'max'),
            order_count=('customer_id', 'count'),
        )

# print first 10 rows
print(
"""
##############
# Exercise 1 #
##############
"""
)

print(output_data.head(10))

##############
# Exercise 2 #
##############

input_data['date_week'] = input_data['date'].apply(lambda x: x.isocalendar()[1])
weekly_data = input_data.groupby(input_data.date_week)['gender'].count()

print(
"""
##############
# Exercise 2 #
##############
"""
)

print(weekly_data.head(5))

weekly_data.plot(kind='bar').get_figure().savefig('weekly.pdf')



##############
# Exercise 3 #
############## 

### plot data by gender
### bins chosen by inspection and repeated plotting
bins = list(range(0, 1500, 50)) # + list(range(1000, 6000, 500))
input_data['product_value'].hist(by=input_data['gender'], alpha = 0.7, bins = bins)[0].get_figure().savefig('genders.pdf')
input_data['product_value'].hist(by=input_data['gender'], alpha = 0.7, bins = bins)[1].get_figure().savefig('genders.pdf')

### failed attempts at plots
# male, female = input_data.groupby('gender')['product_value'].hist(alpha = 0.5)
# input_data['product_value'].plot.hist(by=input_data['gender'], alpha = 0.7, bins = bins).get_figure().savefig("genders.pdf")

### compute mean, std, min, max of product value by gender
gender_data = \
    input_data.groupby('gender') \
        .agg(
            mean_order_value=('product_value', 'mean'),
            standard_error=('product_value', 'sem'),
            standard_deviation=('product_value', 'std'),
            min_order_value=('product_value', 'min'),
            max_order_value=('product_value', 'max'),
            # number_of_orders=('product_value', 'count'),
        )

### output values
mean_0 = gender_data.loc[0, ('mean_order_value')]
mean_1 = gender_data.loc[1, ('mean_order_value')]

sem_0 = gender_data.loc[0, ('standard_error')]
sem_1 = gender_data.loc[1, ('standard_error')]

delta_means = abs(mean_0 - mean_1)

sigma_0 = delta_means / sem_0
sigma_1 = delta_means / sem_1

print(
"""
##############
# Exercise 3 #
##############
"""
)

print(gender_data.head(2))

string = "\nThe difference in mean order value between the two genders of {0:.2f} is significant because it is {1:.2f} ({2:.2f}) times the standard error on the mean of {3:.2f} ({4:.2f}) for gender 0 (1). \nMeans and standard errors were calculated in the standard manner."

# ({.2f} for gender 0 and {.2f} for gender 1) mean_0, mean_1, 

string = string.format(delta_means, sigma_0, sigma_1, sem_0, sem_1)

print(string)

##############
# Exercise 4 #
##############

print(
"""
##############
# Exercise 4 #
##############
"""
)