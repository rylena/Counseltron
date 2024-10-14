import nltk
import numpy as np
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random

nltk.download("punkt")
nltk.download("wordnet")

# File path to the text data
filepath = r'counsel5.txt'
corpus = open(filepath, 'r', errors='ignore')
raw_data = corpus.read()

# Tokenizing the text data
sent_tokens = nltk.sent_tokenize(raw_data)
word_tokens = nltk.word_tokenize(raw_data)

# Initializing the lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Removing punctuation
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Predefined greetings
GREETING_INPUTS = ["hello", "hi", "greetings", "what's up", "hey", "hey there", "yo", "whats good", "namaste"]
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me", "hello my friend"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response, sent_tokens):
    counsellor_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        counsellor_response = "I am sorry! I don't understand you"
    else:
        counsellor_response = sent_tokens[idx]
    
    sent_tokens.pop()  # Remove the user response from sent_tokens
    return counsellor_response

def get_greeting_response(user_response):
    if greeting(user_response) is not None:
        return greeting(user_response)
    else:
        return response(user_response, sent_tokens[:])  # Pass a copy of sent_tokens