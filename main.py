from pathlib import Path
# from File_open_close import file_open_close
import pandas as pd

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
# file_open().info(max_cols=len(file_open())) /Comparing data\
# test1.info(max_cols=len(test1))             \Comparing data/
