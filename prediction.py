import requests
from bs4 import BeautifulSoup
import pickle
import random
from sklearn.naive_bayes import MultinomialNB
import preprocessor as p
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
import re
import os
import nltk
import argparse
if os.name == 'nt':
    import msvcrt
else:
    os.system('pip install getch')
    import getch
nltk.download('wordnet')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def nitter():
    r = requests.get('https://github.com/zedeus/nitter/wiki/Instances')
    soup = BeautifulSoup(r.text, 'html.parser')
    markdownBody = soup.find(class_='markdown-body')
    tables = markdownBody.find_all('table')
    tables.pop(3)
    tables.pop(3)
    nitterList = []
    for table in tables:
        tbody = table.find('tbody')
        trs = tbody.find_all('tr')
        for tr in trs:
            td = tr.find('td')
            a = td.find('a')
            url = a.contents[0]
            if not url.endswith('.onion'):
                url = 'https://' + url
                nitterList.append(url)
    return nitterList


def predict_party(dataf, model):
    return model.predict(dataf)


def clean_data(tweet):
    def preprocess_data(data):
        data = data.replace('\d+', '')  # Removes Numbers
        lower_text = data.lower()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        w_tokenizer = TweetTokenizer()

        def lemmatize_text(text):
            return [(lemmatizer.lemmatize(w)) for w
                    in w_tokenizer.tokenize((text))]

        def remove_punctuation(words):
            new_words = []
            for word in words:
                new_word = re.sub(r'[^\w\s]', '', (word))
                if new_word != '':
                    new_words.append(new_word)
            return new_words

        words = lemmatize_text(lower_text)
        words = remove_punctuation(words)
        return words

    tweet_clean = p.clean(tweet)
    pre = preprocess_data(tweet_clean)
    tweet_clean = pre
    tweet_clean = [item for item in tweet_clean if item not in stop_words]
    return " ".join(tweet_clean)


def start(model, setup, instances):
    os.system('cls' if os.name == 'nt' else 'clear')

    if not model:
        print("Loading model, please wait...")
        with open(setup.model_type + '.pkl', 'rb') as fid:
            model = pickle.load(fid)
    if setup.mode == 2 and len(instances) == 0:
        instances = nitter()

    if setup.mode == 1:
        text = input(
            "Insert text from a politician's tweet and press enter:\n")
        while not text or len(text) == 0:
            text = input(
                "Wrong input. Please enter a tweet's text and press enter:\n")
    else:
        text = input(
            "Insert a link to a politician's tweet and press enter:\n")
        link = re.search(
            "(?:https?:)?\/\/(?:[A-z]+\.)?twitter\.com\/@?(?P<username>[A-z0-9_]+)\/status\/(?P<tweet_id>[0-9]+)\/?", text)
        while not link or len(link.group(0)) == 0:
            text = input(
                "Wrong input. Please enter a link to a politician's tweet and press enter:\n")
            link = re.search(
                "(?:https?:)?\/\/(?:[A-z]+\.)?twitter\.com\/@?(?P<username>[A-z0-9_]+)\/status\/(?P<tweet_id>[0-9]+)\/?", text)
        print("\nScraping tweet, please wait...")
        first = True
        while first:
            try:
                tweet = requests.get(random.choice(instances) +
                                     "/" + link.group(1) + "/status/" + link.group(2))
                first = False
            except:
                print("Link error. Retrying")
        while not tweet.ok:
            try:
                tweet = requests.get(random.choice(
                    instances) + "/" + link.group(1) + "/status/" + link.group(2))
            except:
                tweet.ok = False
                print("Link error. Retrying")
        soup = BeautifulSoup(tweet.text, 'html.parser')
        error = soup.find(class_='error-panel')
        while error:
            tweet = requests.get(random.choice(
                instances) + "/" + link.group(1) + "/status/" + link.group(2))
            soup = BeautifulSoup(tweet.text, 'html.parser')
            error = soup.find(class_='error-panel')
        text = soup.find(
            class_="tweet-content media-body").get_text(strip=True)

    print("\nThe prediction is loading...\n")
    clean_tweet = clean_data(text)
    dataf = pd.Series([clean_tweet])
    pred = predict_party(dataf, model)
    if pred[0] == 'R':
        pred = "Republican"
    else:
        pred = "Democrat"
    print("The predicted political orientation of the tweet is " + pred + "!")

    print("\nDo you want to make another prediction? (y/n)")
    if os.name == 'nt':
        restart = msvcrt.getche().decode("utf-8")
    else:
        restart = getch.getche().decode("utf-8")
    while not restart and restart.lower() != 'y' and restart.lower() != 'n':
        print("\nWrong input. Please type y or n")
        if os.name == 'nt':
            restart = msvcrt.getche().decode("utf-8")
        else:
            restart = getch.getche().decode("utf-8")
    if restart == 'y':
        start(model, setup, instances)
    else:
        return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Predict the political leaning of a tweet.')
    parser.add_argument(
        '-mode', type=int, help="type 1 to use tweet texts (default) or type 2 to input tweet links", default=1, choices=[1, 2])
    parser.add_argument('-model_type', type=str,
                        help="type mnb to use the Multinomial Naive Bayes model or type sgd to use the SGD-trained model", default="mnb", choices=['mnb', 'sgd'])
    args = parser.parse_args()

    start(None, args, [])
