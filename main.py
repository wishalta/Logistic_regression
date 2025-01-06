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

'''Exploratory Data Analysis and Visualization'''


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