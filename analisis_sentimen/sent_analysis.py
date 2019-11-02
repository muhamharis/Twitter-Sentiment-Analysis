import nltk
import random
import string
import re
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
import pickle

class Classify(ClassifierI):
    def __init__(self, classifiers):
        self.classifiers = classifiers

        # classify the most appropriate label for the given features

    def classify(self, features):
        v = self.classifiers.classify(features)
        return v

# opening datasets

pos_data = open('dataset/positive.txt', 'r').read()
neg_data = open('dataset/negative.txt', 'r').read()

documents = []

# adding review and its label to a 'documents' list (output: list of tuples)
for r in pos_data.split('\n'):
    documents.append((r, 'pos'))

for r in neg_data.split('\n'):
    documents.append((r, 'neg'))

all_words = []

# tokenize sentence to words
pos_data_words = word_tokenize(pos_data)
neg_data_words = word_tokenize(neg_data)

stop_words = set(stopwords.words('english'))

# remove stopwords, punctuation and normalize words and add it all to 'all_words'
for w in pos_data_words:
    if w not in stop_words and w not in string.punctuation:
        all_words.append(w.lower())

for w in neg_data_words:
    if w not in stop_words and w not in string.punctuation:
        all_words.append(w.lower())

# remove numbers
for w in all_words:
    w = re.sub('[^a-zA-Z]', ' ', w)

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for w in words:
        lemma = lemmatizer.lemmatize(w, pos='v')
        lemmas.append(lemma)
    return lemmas


all_words = lemmatize_verbs(all_words)


# reduce all_words to contains only most common words (convert words to features) (output: keys:values - words:count)
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())


# function to find features in a document (feature extraction: BoW - unigram)
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# add every feature and its category to featuresets
featuresets = []

for (rev, category) in documents:
    rev = lemmatize_verbs(word_tokenize(rev))
    rev = (' '.join(rev)).strip()
    featuresets.append((find_features(rev), category))

random.shuffle(featuresets)


# 70:30 ratio of 10664 data
training_set = featuresets[:7465]
testing_set = featuresets[7465:]

classifier = SklearnClassifier(LogisticRegression(solver='lbfgs')).train(training_set)

save_classifier = open("pickled/logisticreg.pickle", "wb")
pickle.dump(classifier, save_classifier, pickle.HIGHEST_PROTOCOL)
save_classifier.close()

print('Logistic Regression Accuracy: ', (nltk.classify.accuracy(classifier, testing_set)) * 100)
voted_labels = Classify(classifier)
print("Classification:", voted_labels.classify(testing_set[0][0]))
