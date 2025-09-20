import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse # Needed to save the sparse matrix

# Fetch the data from data/raw folder
train_data = pd.read_csv("./data/raw/train.csv")
test_data  = pd.read_csv("./data/raw/test.csv")

# Download NLTK stopwords and create a set for faster lookups
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Text Preprocessing Functions

def preprocess_text(text):
    """
    A single function to clean and preprocess text.
    - Converts to string and lowercase
    - Removes HTML tags, URLs, and emails
    - Removes punctuation
    - Removes stopwords
    """
    # Ensure it's a string and convert to lowercase
    text = str(text).lower()
    
    # Remove HTML, URLs, and emails
    text = re.sub(r"<.*?>|http\S+|www\.\S+|\S+@\S+", "", text)
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove non-ASCII characters (optional, but good practice)
    text = text.encode("ascii", errors="ignore").decode()
    
    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    # Join back into a single string
    return " ".join(filtered_words)

# Applying Preprocessing 
# Apply the function to the text columns in both dataframes
for df in [train_data, test_data]:
    df['title'] = df['title'].apply(preprocess_text)
    df['text'] = df['text'].apply(preprocess_text)

# Combine the cleaned title and text columns into a new column
train_data['combined_text'] = train_data['title'] + ' ' + train_data['text']
test_data['combined_text']  = test_data['title'] + ' ' + test_data['text']

# Initialize the CountVectorizer
# You can tune parameters like max_features to limit the vocabulary size
vectorizer = CountVectorizer(max_features=500, min_df=2)

# Fit the vectorizer on the TRAINING data and transform it
# This learns the vocabulary from the training data
X_train_vectorized = vectorizer.fit_transform(train_data['combined_text'])

# This ensures the test data is vectorized using the same vocabulary
X_test_vectorized = vectorizer.transform(test_data['combined_text'])


# Storing the Processed Data 
# Define the path for processed data
data_path = os.path.join("data", "processed")

# Create the directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)

# Save the dataframes with the new combined_text column
train_data.to_csv(os.path.join(data_path, "train_processed_text.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test_processed_text.csv"), index=False)

# Save the vectorized sparse matrices
# These are the files you will load fcor model training
scipy.sparse.save_npz(os.path.join(data_path, "train_vectorized.npz"), X_train_vectorized)
scipy.sparse.save_npz(os.path.join(data_path, "test_vectorized.npz"), X_test_vectorized)