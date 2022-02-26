import re
import string
import pandas as pd
import requests
import numpy as np
import nltk
import joblib
import sys
from random import shuffle
from nltk.corpus import twitter_samples
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from nltk import data

message = sys.argv[1]

log_folder = './'

data.path.append(log_folder)
nltk.download('twitter_samples', download_dir = log_folder)
nltk.download('stopwords', download_dir = log_folder)

pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')
print(f"positive sentiment GOOD total samples {len(pos_tweets)}")
print(f"negative sentiment  Bad total samples {len(neg_tweets)}")

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
processed_pos_tweets = text_processor.preprocess(pos_tweets)
processed_neg_tweets = text_processor.preprocess(neg_tweets)

def build_bow_dict(tweets, labels):
    freq = {}
    for tweet, label in list(zip(tweets, labels)):
        for word in tweet:
            freq[(word, label)] = freq.get((word, label), 0) + 1    
    return freq

labels = [1 for i in range(len(processed_pos_tweets))]
labels.extend([0 for i in range(len(processed_neg_tweets))])

twitter_processed_corpus = processed_pos_tweets + processed_neg_tweets
bow_word_frequency = build_bow_dict(twitter_processed_corpus, labels)

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

print(formData)

print('Your input sentence is: ' + message)
res = requests.post('http://127.0.0.1:8081/v1/models/model:predict', json=formData)
print(res.text)
