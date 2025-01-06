from pathlib import Path
from File_open_close import file_open_close
import pandas as pd

# FILES_PATH = Path("2_logistic_regression_files/")
# FILES_PATH.mkdir(parents=True, exist_ok=True)

raw_df = pd.read_csv("weatherAUS.csv")
tear = file_open_close(raw_df)
print(tear)

read = tear.file_interact()
file = pd.DataFrame(read)
print(file)

