from pathlib import Path
# from File_open_close import file_open_close
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame

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
    print(sampled_raw_df)
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
sns.countplot(x=pd.to_datetime(sampled_raw_df.Date).dt.year)  # <---- x=pd.to_datetime(sampled_raw_df.Date) x kintamasis kuriam suteikiamos aplamai visos datos
                                                                # dt.year <---- paima tik metus, taip ir sudaroma lentele is kiekvienu metu duomenu kiekio
# plt.show()

year = pd.to_datetime(sampled_raw_df.Date).dt.year # Perfect example, indexes mixed because sample took 10% of data randomly
print(year)

train_df = sampled_raw_df[year < 2015]
val_df = sampled_raw_df[year == 2015]
test_df = sampled_raw_df[year > 2015]

# print('train_df.shape :', train_df.shape)      # /
# print('val_df.shape :', val_df.shape)          #| <---- Shape tells how many rows and columns are in the dataset.
# print('test_df.shape :', test_df.shape)        # \

plt.title('Months Validation test')
sns.countplot(x=pd.to_datetime(val_df.Date).dt.month)
plt.show()

'''NO TEST''' # While not a perfect 60-20-20 split, we have ensured that the test validation and test sets both contain data for all 12 months of the year.
