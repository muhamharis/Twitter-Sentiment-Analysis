from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from nltk.tokenize import WordPunctTokenizer
import re
import import_ipynb
import sentiment_mod as s


# consumer key, consumer secret, access token, access secret.
ckey = ""
csecret = ""
atoken = "-"
asecret = ""


class Listener(StreamListener):

    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            try:
                tweet = status.retweeted_status.extended_tweet["full_text"]
            except:
                tweet = status.retweeted_status.text
        else:
            try:
                tweet = status.extended_tweet["full_text"]
            except AttributeError:
                tweet = status.text

        cleaned_tweets = clean_tweets(tweet)
        sentiment_value = s.sentiment(cleaned_tweets)
        print(cleaned_tweets, sentiment_value)

        output = open("twitter-out.txt", "a")
        output.write(sentiment_value)
        output.write('\n')
        output.close()
        return True

    def on_error(self, status):
        print(status)

def clean_tweets(tweet):
    rt_removed = re.sub('RT @[\w_]+: ', '', tweet)
    user_removed = re.sub(r'@[A-Za-z0-9]+', '', rt_removed)
    link_removed = re.sub('https?://[A-Za-z0-9./]+', '', user_removed)
    number_removed = re.sub('[^a-zA-Z]', ' ', link_removed)
    lower_case_tweet = number_removed.lower()
    tok = WordPunctTokenizer()
    words = tok.tokenize(lower_case_tweet)
    words = s.lemmatize_verbs(words)
    clean_tweet = (' '.join(words)).strip()
    return clean_tweet


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, Listener())
twitterStream.filter(track=["Spider-Man: Far From Home"], languages=["en"])
