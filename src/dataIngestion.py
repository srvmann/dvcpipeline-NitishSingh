import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Importing the fake news dataset 
fake_news_original = pd.read_csv(r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Datasets\Fake-Real-News\Fake.csv")
fake_news_original["label"] = "fake"

# Importing the true news dataset 
true_news_original = pd.read_csv(r"C:\Users\Varun\Downloads\Saurav\Outsource 360 internship\Project\Datasets\Fake-Real-News\True.csv")
true_news_original["label"] = "true"

# Combining both the datasets
df = pd.concat([true_news_original,fake_news_original],ignore_index = True)

# Converting the date column into actual date Dtype column from object Dtype
df["date"] = pd.to_datetime(df["date"], errors = 'coerce', dayfirst = True, format = 'mixed')

# Droping the rows date rows which holds link [Drop rows where 'date' is NaT]
df = df.dropna(subset=['date'])

# Replacing the label values {True : 1,Fake : 0} 
df["label"].replace({"fake" : 0,"true" : 1},inplace = True)

# Splitting the data in train.csv and test.csv to avoid any kind of data leakage
train_data , test_data = train_test_split(df,stratify = df["label"],test_size = 0.2,random_state = 33)
 
# Storing the splitted data in data folder inside which raw is being created
data_path = os.path.join("data","raw")

# creating the directory 
os.makedirs(data_path)

# saving the dataset{train and test}
train_data.to_csv(os.path.join(data_path,"train.csv"),index = False )
test_data.to_csv(os.path.join(data_path,"test.csv"),index = False)