from pathlib import Path
# from File_open_close import file_open_close
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
import numpy as np
from seaborn.external.husl import dot_product

FILES_PATH = Path("2_logistic_regression_files/")
FILES_PATH.mkdir(parents=True, exist_ok=True)

def file_open():
    raw_df = pd.read_csv("weatherAUS.csv")
    return raw_df

'''We are trying to predict about tomorrow, will it rain or not?'''

# print(file_open())
# file_open().info(max_cols=len(file_open()))

test1 = file_open()                                            #<----Copy to not lose first data
test1.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True) #<----#Good idea to discard the rows where the value of RainTomorrow or
                                                                       #RainToday is missing to make our analysis and modeling simpler (since one of them is the target
                                                                       #variable, and the other is likely to be very closely related to the target variable).
# file_open().info(max_cols=len(file_open())) #/Comparing data\
# test1.info(max_cols=len(test1))             #\Comparing data/
'''



                                 ''''''Exploratory Data Analysis and Visualization''''''
                                 
                                 
                                 
'''
import plotly.express as px           # plotly excels at interactive plots, while matplotlib and seaborn are ideal for static plots.
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')     # <---- Changes the visual style of plots to a dark grid background.
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

magic = px.histogram(test1, x="Location", title="Location vs. Rainy Days", color="RainToday")
# magic.show()


magic1 = px.histogram(test1,
             x='Temp3pm',
             title='Temperature at 3 pm vs. Rain Tomorrow',
             color='RainTomorrow')
# magic1.show()

magic2 = px.scatter(test1.sample(2000),
           title='Min Temp. vs Max Temp.',
           x='MinTemp',
           y='MaxTemp',
           color='RainToday')
# magic2.show()

magic3 = px.scatter(test1.sample(2000),
           title='Temp (3 pm) vs. Humidity (3 pm)',
           x='Temp3pm',
           y='Humidity3pm',
           color='RainTomorrow')
# magic3.show()

magic4 = px.scatter_matrix(test1, height=2000, color='RainTomorrow', opacity=0.1).update_traces(marker=dict(size=1))
# magic4.show()

'''



                                 ''''''(Optional) Working with a Sample''''''
                                 
                                 
                                 
'''
USE_SAMPLE = True      # <---- A subset of the data will be selected. If False it uses full dataset
sample_fraction = 0.1  # <---- The fraction of the data to sample (in this case, 10%).
if USE_SAMPLE:
    sampled_raw_df = test1.sample(frac=0.1).copy()   # <---- Makes copy
    # print(sampled_raw_df)
'''



                                 ''''''Training, Validation and Test Sets''''''
                                 
                                 
                                 
'''
# 1.Training set - used to train the model, i.e., compute the loss and adjust the model's weights using an optimization technique.
#
# 2.Validation set - used to evaluate the model during training, tune model hyperparameters (optimization technique, regularization etc.),
# and pick the best version of the model. Picking a good validation set is essential for training models that generalize well. Learn more here.
#
# 3.Test set - used to compare different models or approaches and report the model's final accuracy. For many datasets, test sets are
# provided separately. The test set should reflect the kind of data the model will encounter in the real-world, as closely as feasible.

'''EXAMPLE IN SECOND PAGE'''

plt.title('No. of Rows per Year')
sns.countplot(x=pd.to_datetime(test1.Date).dt.year)  # <---- x=pd.to_datetime(sampled_raw_df.Date) x kintamasis kuriam suteikiamos aplamai visos datos
                                                                # dt.year <---- paima tik metus, taip ir sudaroma lentele is kiekvienu metu duomenu kiekio
# plt.show()

year = pd.to_datetime(test1.Date).dt.year # Perfect example, indexes mixed because sample took 10% of data randomly
# print(year)

train_df = test1[year < 2015]
val_df = test1[year == 2015]
test_df = test1[year > 2015]

# print('train_df.shape :', train_df.shape)      # /
# print('val_df.shape :', val_df.shape)          #| <---- Shape tells how many rows and columns are in the dataset.
# print('test_df.shape :', test_df.shape)        # \

plt.title('Months Validation test')
sns.countplot(x=pd.to_datetime(val_df.Date).dt.month)
# plt.show()

'''NO TEST''' # While not a perfect 60-20-20 split, we have ensured that the test validation and test sets both contain data for all 12 months of the year.

'''



                                 ''''''Identifying Input and Target Columns''''''



'''
# Often, not all the columns in a dataset are useful for training a model. In the current dataset, we can ignore the Date
# column, since we only want to weather conditions to make a prediction about whether it will rain the next day.
input_cols = list(train_df.columns)[1:-1]  # <---- columns is an attribute of the DataFrame that returns an Index object containing the names of all the columns in the DataFrame.
target_col = 'RainTomorrow'
print(input_cols)
print(target_col)

X_train = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

X_val = val_df[input_cols].copy()
y_val = val_df[target_col].copy()

X_test = test_df[input_cols].copy()
y_test = test_df[target_col].copy()

# print(X_train.head())
# print(train_targets.value_counts())
# print(X_train.info(max_cols=X_train.shape[1]))  # <---- identify which of the columns are numerical and which ones are categorical (object, float64)


numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
# The select_dtypes() function filters the DataFrame based on the data type of its columns.
# include=np.number selects columns that have a numeric data type (e.g., integers, floats).
# .columns:   <-----   This retrieves the names of the columns in the filtered DataFrame.
# .tolist():  <-----   Converts the column names (a Pandas Index object) into a regular Python list.
categorical_cols = X_train.select_dtypes('object').columns.tolist()
# It takes object data types
# print(numeric_cols)
# print(categorical_cols)
# print(X_train[numeric_cols].describe()) # describe <---- adds number after dot
# print(X_train[categorical_cols].nunique()) # The number of unique values in categorical columns
'''



                                 ''''''Imputing Missing Numeric Data''''''



'''

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'mean')
imputer2 = SimpleImputer(strategy = 'median')

# print(test1[numeric_cols].isna().sum()) # Takes all columns with numbers and counts how may NUN values are there

test2 = test1#X_train[numeric_cols]
# print(test2)
lol = px.scatter( test1["Sunshine"],test2["Date"], opacity=0.2).update_traces(marker=dict(size=3))        # Works, but meaning can't imagine
# lol.show()
lol1 = px.scatter( test1["Sunshine"], opacity=0.2).update_traces(marker=dict(size=3))
# lol1.show()
# print(X_train[numeric_cols].isna().sum())

imputer.fit(test1[numeric_cols])
'''
fit() is a method used to analyze the data and compute the necessary statistics needed for imputation.
For example:
If using SimpleImputer with strategy='mean', .fit() calculates the mean of each column.
If using strategy='median', it computes the median for each column.
These computed values are stored internally in the imputer object and will later be used during the transform step to replace missing values.
'''
# print(list(imputer.statistics_))      # <---- with which numbers filled emtpy gaps

X_train[numeric_cols] = imputer.transform(X_train[numeric_cols])
X_val[numeric_cols] = imputer.transform(X_val[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

# print(X_train[numeric_cols].isna().sum())    # <---- shows are there empty spaces

# abc1 = px.scatter(test1["Sunshine"], opacity=0.2).update_traces(marker=dict(size=3))
# abc1.show()
# abc2 = px.scatter(X_train["Sunshine"], opacity=0.2).update_traces(marker=dict(size=3))
# abc2.show()
'''



                                 ''''''Scaling Numeric Features''''''



'''
# print(test2[numeric_cols[:-1]].describe())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train[numeric_cols])

# print(f'Minimum: {list(scaler.data_min_)}')
# print(f'Maximum: {list(scaler.data_max_)}')

X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])       # keiciam skaitinius stulpelius su skaitiniais stulpeliais
X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# print(X_train[numeric_cols].describe())

board = px.scatter(test1[["WindSpeed9am"]])
# board.show()
board1 = px.scatter(X_train[["WindSpeed9am"]])
# board1.show()
board2 = px.scatter(X_test[["WindSpeed9am"]])
# board2.show()
'''



                                 ''''''Encoding Categorical Data''''''



'''
print(test1[categorical_cols].nunique()) # categorical_cols <--- stulpeliu pavadinimai

from sklearn.preprocessing import OneHotEncoder  # tik sita funkcija sklearn, nes jai import sklearn atsisiustu per daug nereikalingo stuff

# print(categorical_cols)
#The handle_unknown='ignore' parameter in the OneHotEncoder ensures that if a category appears in the test set
# (or any data during transformation) that was not present in the training set, it will be ignored instead of raising an error.
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # is pradziu be ignore, paziuret ar nera problemu, jei yra bet nesvarbus tai ignore
encoder.fit(test1[categorical_cols])
# print(encoder.categories_)


encoded_cols = list(encoder.get_feature_names_out(categorical_cols))  #sudaro daugiau column pavadinimu, su visais miestais atskirai, kryptim ir t.t.
# print(encoded_cols) # test1[categorical_cols].nunique() <--- is cia paima unique items

X_train.loc[:, encoded_cols] = encoder.transform(X_train[categorical_cols])
X_val.loc[:,encoded_cols] = encoder.transform(X_val[categorical_cols])
X_test.loc[:,encoded_cols] = encoder.transform(X_test[categorical_cols])
'''



                                 ''''''Training a Logistic Regression Model''''''



'''
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
# print(model.fit(X_train[numeric_cols], train_targets))
print('-----------------------------')
model = LogisticRegression(solver="liblinear")
# print(model.fit(X_train[numeric_cols + encoded_cols], train_targets))
print('-----------------------------')
# print(numeric_cols + encoded_cols)
print('-----------------------------')
# print(model.coef_.tolist()[0])
print('-----------------------------')
# print(model.intercept_)
'''



                                 ''''''Making Predictions and Evaluating the Model''''''



'''
X_train = X_train[numeric_cols + encoded_cols]
X_val = X_val[numeric_cols + encoded_cols]
X_test = X_test[numeric_cols + encoded_cols]

# train_preds = model.predict(X_train) #yes and no returns
# print(train_preds)
#
# train_probs = model.predict_proba(X_train) # % prediction
# print(train_probs)

'''To be continued...'''