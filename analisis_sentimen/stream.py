from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s

# consumer key, consumer secret, access token, access secret.
ckey = "5IXPaQbji48SxxliDTMCrIPGd"
csecret = "YLajBfxUignofLq1wLdb0wohDubwkG8iMzZhxxvV5fYwSoNDn0"
atoken = "1064160392354529282-pDeaKK9rYzIgRSfdLFdUw1y7ryu5P8"
asecret = "HwpMgyyNHd0vze5cpXfIRb3L3abdL50DJguJouqdyMUED"


class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        sentiment_value = s.sentiment(tweet)
        print(tweet, sentiment_value)

        output = open("twitter-out.txt", "a")
        output.write(sentiment_value)
        output.write('\n')
        output.close()

        return True

    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])
