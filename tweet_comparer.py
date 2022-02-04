import ast
import requests
import spacy
import en_core_web_sm
import numpy as np
from sklearn.linear_model import LogisticRegression

nlp = en_core_web_sm.load()


def get_userdata(username):
    '''gets twitter user data for use in get_tweets.  
    Must be twitter handle as a string, no @ character.'''

    status = requests.get(
        f"https://bloomtech-ds-twit-assist.herokuapp.com/user/{username}")
    userdata = ast.literal_eval(status.text)
    return userdata


def get_tweets(self):
    '''Turns the literal string dicts into spacey-usable text lists of the tweets.
    User should be a literal string translation of returned data from the twitter 
    api.
    Use: Newvarname = get_tweets(User).'''

    tweetlist = []

    for x in range(len(self['tweets'])):
        addtweet = self['tweets'][x]['full_text']
        tweetlist.append(addtweet)
    return tweetlist


def tweets_to_vector(self):
    '''Turns a list consisting of strings into an array of vectors.
    Must define your spacy NLP model as nlp.'''

    vects = np.array([nlp(tweet).vector for tweet in self])
    return vects


def make_model(user1, user2):
    '''Makes predictive model from two twitter users.
    Used to simplify final predict function.
    Requires 2 twitter handles as a string, no @ character.'''

    userdata1 = get_userdata(user1)
    tweets1 = get_tweets(userdata1)
    vectors1 = tweets_to_vector(tweets1)

    userdata2 = get_userdata(user2)
    tweets2 = get_tweets(userdata2)
    vectors2 = tweets_to_vector(tweets2)

    allvectors = np.vstack([vectors1, vectors2])
    labels = np.concatenate([np.zeros(len(tweets1)), np.ones(len(tweets2))])

    model = LogisticRegression().fit(allvectors, labels)
    return model


def predict(user1, user2, sample_text):
    '''Predict which user a test tweet belongs to.
    Requires 2 twitter handles as a string, no @ character,
    and a sample text as a string.'''

    model = make_model(user1, user2)
    sample_vectorized = (nlp(sample_text).vector).reshape(1, -1)
    prediction = model.predict(sample_vectorized)
    if prediction == [0.]:
        return f'This tweet is probably by {user1}'
    elif prediction == [1.]:
        return f'This tweet is probably by {user2}'
    else:
        return 'There was an error, please try again.'


# combining short texts to make the long input text inbounds
shorttext1 = 'Enter the two users you wish to compare, followed by the text '
shorttext2 = 'you wish to test, each seperated by ",":'
longtext = shorttext1 + shorttext2

x, y, z = input(longtext).split(',')

print(predict(x, y, z))
# Use these values to test:
# elonmusk, CoreyMSchafer, Today's pandas tutorial is now uploaded!
# if error occurs when making array, just run again.
