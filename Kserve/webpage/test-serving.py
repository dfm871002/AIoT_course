import re
import sys
import string
import pandas as pd
import requests
from random import shuffle
import torch
import numpy as np
import nltk
import joblib
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import data

log_folder = './'

message = sys.argv[1]

bow_word_frequency = joblib.load(open(log_folder + '/bow_word_frequency.pkl','rb'))

class Preprocess():   
    def __init__(self):
        self.tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,reduce_len=True)
        self.stopwords_en = stopwords.words('english') 
        self.punctuation_en = string.punctuation
        self.stemmer = PorterStemmer()        
    def __remove_unwanted_characters__(self, tweet):
        tweet = re.sub(r'^RT[\s]+', '', tweet)
        tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
        tweet = re.sub(r'#', '', tweet)
        tweet = re.sub('\S+@\S+', '', tweet)
        tweet = re.sub(r'\d+', '', tweet)
        return tweet    
    def __tokenize_tweet__(self, tweet):        
        return self.tokenizer.tokenize(tweet)   
    def __remove_stopwords__(self, tweet_tokens):
        tweets_clean = []
        for word in tweet_tokens:
            if (word not in self.stopwords_en and word not in self.punctuation_en):
                tweets_clean.append(word)
        return tweets_clean   
    def __text_stemming__(self,tweet_tokens):
        tweets_stem = [] 
        for word in tweet_tokens:
            stem_word = self.stemmer.stem(word)  
            tweets_stem.append(stem_word)
        return tweets_stem
    def preprocess(self, tweets):
        tweets_processed = []
        for _, tweet in tqdm(enumerate(tweets)):        
            tweet = self.__remove_unwanted_characters__(tweet)            
            tweet_tokens = self.__tokenize_tweet__(tweet)            
            tweet_clean = self.__remove_stopwords__(tweet_tokens)
            tweet_stems = self.__text_stemming__(tweet_clean)
            tweets_processed.extend([tweet_stems])
        return tweets_processed

def extract_features(processed_tweet, bow_word_frequency):
    features = np.zeros((1,3))
    features[0,0] = 1
    for word in processed_tweet:
        features[0,1] = bow_word_frequency.get((word, 1), 0) + features[0,1]
        features[0,2] = bow_word_frequency.get((word, 0), 0) + features[0,2]
    return features

text_processor = Preprocess()

data = [message]
data = text_processor.preprocess(data)
            
data_o = str(data)
data_o = data_o[2:len(data_o)-2]

vect = np.zeros((1, 3))
for index, tweet in enumerate(data):
    vect[index, :] = extract_features(tweet, bow_word_frequency)

formData = {
    'instances': vect.tolist()
}

res = requests.post('http://127.0.0.1:8081/v1/models/model:predict', json=formData)
if "0" in res.text:
  print("It's Negative Sentiment.")
else:
  print("It's Positive Sentiment.")
