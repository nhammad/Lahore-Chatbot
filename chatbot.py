import nltk
import string
import io
import random
import sys
import warnings 
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
warnings.filterwarnings('ignore')

# coding=<encoding name>

f=open('lahore.txt','r',encoding='utf8', errors='ignore')
raw = f.read()
raw = raw.lower() # converts to lowercase


#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only
nltk.download('popular', quiet=True) # for downloading packages 

# Tokenization: 
sent_tokens = nltk.sent_tokenize(raw) # converts to list of sentences 
word_tokens = nltk.word_tokenize(raw) # converts to list of words

# Pre-processing the raw text
# WordNet is a semantically-oriented dictionary of English included in NLTK.

lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching:
# If a users input is a greeting, the bot shall return a greeting response

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hello mate!", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]    
    if(req_tfidf == 0):
        robo_response = robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

flag=True
print("CHATTER: My name is CHATTER. I will answer your queries about Lahore and its history. If you want to exit, type Bye!")
while(flag == True):
    user_response = input("Me: ")
    user_response=user_response.lower()
    if(user_response !='bye'):
        if(user_response == 'thanks' or user_response == 'thank you' ):
            flag=False
            print("CHATTER: You are welcome..")
        else:
            if(greeting(user_response) != None):
                print("CHATTER: " +greeting(user_response))
            else:
                print("CHATTER: ", end = "")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("CHATTER: Bye! take care..")
