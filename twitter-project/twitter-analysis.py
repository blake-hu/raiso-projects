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

# Looking up and replacement
print('Looking up and replacement')
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: x.lower())
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: lookup_dict(x,SHORT_WORD_DICT))
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: lookup_dict(x,APOSTROPHE_DICT))
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: lookup_dict(x,EMOTICON_DICT))

# Replacing punctuation and numbers
print('Replacing punctuation and numbers')
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
train_df['clean_tweet'] = train_df['clean_tweet'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))

# Remove if this takes too much time
print('Remove if this takes too much time')
train_df['clean_tweet'].apply(lambda x: str(TextBlob(x).correct()))


# Creating token for the clean tweets
print('Creating token for the clean tweets')
from nltk.tokenize import word_tokenize
nltk.download('punkt')
train_df['tweet_token'] = train_df['clean_tweet'].apply(lambda x: word_tokenize(x))

# Importing stop words from NLTK coupus and word tokenizer
print('Importing stop words from NLTK coupus and word tokenizer')
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
train_df['tweet_token_filtered'] = train_df['tweet_token'].apply(lambda x: [word for word in x if not word in stop_words])

print(train_df)


# Importing library for stemming
print('Importing library for stemming')
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
train_df['tweet_stemmed'] = train_df['tweet_token_filtered'].apply(lambda x: ' '.join([stemming.stem(i) for i in x]))

from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizing = WordNetLemmatizer()
train_df['tweet_lemmatized'] = train_df['tweet_token_filtered'].apply(lambda x: ' '.join([lemmatizing.lemmatize(i) for i in x]))
train_df['tweet_lemmatized'].head(10)
print(train_df)