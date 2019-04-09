import random
import nltk
import pickle
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize


class Classify(ClassifierI):
    def __init__(self, classifiers):
        self.classifiers = classifiers

    # classify the most appropriate label for the given features
    def classify(self, features):
        v = self.classifiers.classify(features)
        return v


documents_f = open('pickled/documents.pickle', 'rb')
documents = pickle.load(documents_f)
documents_f.close()

word_features_f = open('pickled/word_features.pickle', 'rb')
word_features = pickle.load(word_features_f)
word_features_f.close()


# function to find features in a document
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets_f = open('pickled/featuresets.pickle', 'rb')
featuresets = pickle.load(featuresets_f)
featuresets_f.close()

random.shuffle(featuresets)

# 70:30 ratio of 10664 data
training_set = featuresets[:7465]
testing_set = featuresets[7465:]

classifier_f = open('pickled/naivebayes.pickle', 'rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

text_classifier = Classify(classifier)

print('Naive Bayes Accuracy: ', (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)


def sentiment(text):
    feats = find_features(text)
    return text_classifier.classify(feats)
