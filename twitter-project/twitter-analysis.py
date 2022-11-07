# Importing Libraries 
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from textblob import TextBlob
from twitter_constants import *

train_df  = pd.read_csv('data/twitter_subset.csv')
test_df = pd.read_csv('data/twitter_validation.csv')

def lookup_dict(text, dictionary):
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text

# Importing HTMLParser
from html.parser import HTMLParser
html_parser = HTMLParser()

train_df['clean_tweet'] = train_df['message'].apply(lambda x: html_parser.unescape(x))
print(train_df)

# TODO: Try using lambda functions on your own now!
print('Looking up and replacement')

# Replacing punctuation and numbers
print('Replacing punctuation and numbers')
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))


# Creating token for the clean tweets
print('Creating token for the clean tweets')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
# TODO: Try using lambda functions and applying word_tokenize to the words!

# Importing stop words from NLTK coupus and word tokenizer
print('Importing stop words from NLTK coupus and word tokenizer')
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
#  TODO: How do you remove stopwords using a lambda function?

print(train_df)


# Importing library for stemming
print('Importing library for stemming')
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
# TODO: Use PorterStemmer.stem() to create a new tweet_stem column! All words must be joined by a space.
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizing = WordNetLemmatizer()
# TODO: Use WordNetLemmatizer.lemmatize() to create a new tweet_lemmatized column!
train_df['tweet_lemmatized'].head(10)
print(train_df)