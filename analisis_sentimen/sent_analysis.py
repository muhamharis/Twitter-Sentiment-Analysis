import nltk
import random
import string
from nltk.corpus import movie_reviews, stopwords

# tokenizing each file to words and label them with its category for training (output: list of tuples)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# shuffle the data to avoid ordered label
random.shuffle(documents)

stop_words = set(stopwords.words('english'))

# remove stopwords, punctuation and normalize movie_review corpus and add it to all_words
all_words = [w.lower() for w in movie_reviews.words() if w not in stop_words and w not in string.punctuation]

# reduce all_words to contains only most common words (convert words to features) (output: keys:values - words:count)
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())


# function to find features in a document
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# add every feature and its category to featuresets
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# 70:30 ratio of 2000 data
training_set = featuresets[:1400]
testing_set = featuresets[1400:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print('Naive Bayes Accuracy: ', (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)
