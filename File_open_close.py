import pandas as pd
#
# class file_open_close():
#     def __init__(self, raw_df):
#         self.raw_df = raw_df
#
#     def file_interact(self):
#         self.file_open()
#         self.file_close()
#
#     def file_open(self):
#         self.raw_df = pd.read_csv("weatherAUS.csv")
#         return self.raw_df
#
#     def file_close(self):
#         self.raw_df.close()

                    '''Training, Validation and Test Sets'''
'''
from sklearn.model_selection import train_test_split

# Sample dataset
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]  # Features
y = [0, 1, 0, 1, 0]  # Labels (target)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)  <---Test size 40% <--- random_state same one don't switch info

# Output the splits
print("Training Features:", X_train)
print("Testing Features:", X_test)
print("Training Labels:", y_train)
print("Testing Labels:", y_test)
'''