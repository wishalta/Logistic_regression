# import pandas as pd
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