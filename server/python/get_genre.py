import csv
import nltk
import math
import pickle
from collections import Counter

import re
import unidecode
import contractions
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')


def get_label(labels):
    for label in labels:
        if label == 'Folk':
            return label
        elif label == 'Hip Hop' or 'rap' in label:
            return 'Rap'
        elif 'K-Pop' in label:
            return 'K-Pop'
        elif 'Rock' in label:
            return 'Rock'
        elif 'Pop' in label:
            return 'Pop'
        elif 'Metal' in label:
            return 'Metal'
        elif 'Funk' in label:
            return 'Funk'
        elif 'R&B' == label:
            return 'R&B'
        elif 'Country' in label:
            return 'Country'
        elif 'Indie' in label:
            return 'Indie'
        elif 'Jazz' in label:
            return 'Jazz'
        return 'OOV'


def tokenize_doc_and_more(text):
    # Implement me!
    bow = Counter()
    punctuation_table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    # tokenize sentences, remove whitespace and set all words to lowercase
    sentences = [re.sub(' +', ' ', sent.lower()) for sent in sent_tokenize(text)]
    # expand contractions, remove punctuation and accents
    sentences = [unidecode.unidecode(contractions.fix(sent).translate(punctuation_table)) for sent in sentences]
    # clear up any newly created whitespace
    sentences = [re.sub(' +', ' ', sent.lower()) for sent in sentences]
    # tokenize words of each sentence
    words = [word_tokenize(word) for word in sentences]
    # flatten list of words, lemmatize them, and then filter stop words
    for word in [lemmatizer.lemmatize(w) for sent in words for w in sent]:
        if word not in stop_words:
            bow.update([word])
    return bow


class NaiveBayes:
    def __init__(self, train_data, test_data, tokenizer, rw1=None, rw2=None, rw3=None, rw4=None):
        # Vocabulary is a set that stores every word seen in the training data
        if rw4:
            with open(rw4, 'rb') as w4f:
                self.vocab = pickle.load(w4f)
        else:
            self.vocab = set()
        self.tokenizer = tokenizer
        self.train_fn = train_data
        self.test_fn = test_data
        if rw2:
            with open(rw2, 'rb') as w2f:
                self.class_total_doc_counts = pickle.load(w2f)
        else:
            self.class_total_doc_counts = {"Folk": 0.0,
                                           "Rap": 0.0,
                                           "Rock": 0.0,
                                           "Pop": 0.0,
                                           "Metal": 0.0,
                                           "Funk": 0.0,
                                           "R&B": 0.0,
                                           "Country": 0.0,
                                           "K-Pop": 0.0,
                                           "Indie": 0.0,
                                           "Jazz": 0.0}
        if rw3:
            with open(rw3, 'rb') as w3f:
                self.class_total_word_counts = pickle.load(w3f)
        else:
            self.class_total_word_counts = {"Folk": 0.0,
                                            "Rap": 0.0,
                                            "Rock": 0.0,
                                            "Pop": 0.0,
                                            "Metal": 0.0,
                                            "Funk": 0.0,
                                            "R&B": 0.0,
                                            "Country": 0.0,
                                            "K-Pop": 0.0,
                                            "Indie": 0.0,
                                            "Jazz": 0.0}
        if rw1:
            with open(rw1, 'rb') as w1f:
                self.class_word_counts = pickle.load(w1f)
        else:
            self.class_word_counts = {"Folk": Counter(),
                                      "Rap": Counter(),
                                      "Rock": Counter(),
                                      "Pop": Counter(),
                                      "Metal": Counter(),
                                      "Funk": Counter(),
                                      "R&B": Counter(),
                                      "Country": Counter(),
                                      "K-Pop": Counter(),
                                      "Indie": Counter(),
                                      "Jazz": Counter()}

    def train_model(self):
        with open(self.train_fn, encoding='utf-8') as csvfile:
            csvf = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvf:
                new_label = get_label(row[1:-1])
                if new_label == "OOV":
                    continue
                text = row[-1]
                self.tokenize_and_update_model(text, new_label)

    def update_model(self, bow, label):
        self.class_total_doc_counts[label] += 1
        for word, count in zip(bow.keys(), bow.values()):
            self.vocab.add(word)
            self.class_word_counts[label].update([word])
            self.class_total_word_counts[label] += count

    def tokenize_and_update_model(self, doc, label):
        self.update_model(self.tokenizer(doc), label)

    def p_word_given_label(self, word, label):
        return self.class_word_counts[label].get(word, 0) / self.class_total_word_counts[label]

    def p_word_given_label_and_alpha(self, word, label, alpha):
        return (self.class_word_counts[label].get(word, 0) + alpha) / \
               (self.class_total_word_counts[label] + alpha * len(self.vocab))

    def log_likelihood(self, bow, label, alpha):
        return sum(math.log(self.p_word_given_label_and_alpha(word, label, alpha)) for word in bow.keys())

    def log_prior(self, label):
        label_docs = self.class_total_doc_counts[label]
        total_docs = sum(self.class_total_doc_counts.values())
        return math.log(label_docs / total_docs)

    def likelihood_ratio(self, word, alpha):
        return self.p_word_given_label_and_alpha(word, 'pos', alpha) / \
               self.p_word_given_label_and_alpha(word, 'neg', alpha)

    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        labels = self.class_word_counts.keys()
        return sorted([(label, self.unnormalized_log_posterior(bow, label, alpha))
                       for label in labels], key=lambda x: -x[1])[0][0]

    def evaluate_classifier_accuracy(self, alpha):
        correct = 0
        total = 0
        with open(self.test_fn, encoding='utf-8') as csvfile:
            csvf = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvf:
                new_label = get_label(row[1:-1])
                if new_label == "OOV":
                    continue
                text = row[-1]
                bow = self.tokenizer(text)
                if self.classify(bow, alpha) == new_label:
                    correct += 1
                total += 1
        return 100 * correct / total


def eval_class(classifier):
    train_acc = classifier.evaluate_classifier('train_data.csv')
    print('acc {}'.format(train_acc*100))


def bag_of_words(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = list(filter(lambda token: token not in stop_words, tokens))
    feats = dict((token, 1) for token in tokens)

    return feats


nb = NaiveBayes(train_data='train_data.csv', test_data='test_data.csv', tokenizer=tokenize_doc_and_more,
                rw1='w1', rw2='w2', rw3='w3', rw4='w4')

# This block is to save the trained model
# nb.train_model()
"""
w1 = nb.class_word_counts
w2 = nb.class_total_doc_counts
w3 = nb.class_total_word_counts
w4 = nb.vocab

with open('w1', 'wb') as w1_file:
    pickle.dump(w1, w1_file)
with open('w2', 'wb') as w2_file:
    pickle.dump(w2, w2_file)
with open('w3', 'wb') as w3_file:
    pickle.dump(w3, w3_file)
with open('w4', 'wb') as w4_file:
    pickle.dump(w4, w4_file)
print(nb.evaluate_classifier_accuracy(0.2))
"""


def classify_from_web(text):
    bow = nb.tokenizer(text)
    return nb.classify(bow, 0.2)


if __name__ == "__main__":
    artist_to_genre = dict()
    lyrics_to_artist = dict()
    lyrics_to_genre = dict()
    genres = set()
    # print(classify_from_web("We were both young when I first saw you I close my eyes and the flashback starts I'm standin' there On a balcony in summer air See the lights, see the party, the ball gowns See you make your way through the crowd And say, Hello Little did I know That you were Romeo, you were throwin' pebbles And my daddy said, Stay away from Juliet And I was cryin' on the staircase Beggin' you, Please don't go,  and I said Romeo, take me somewhere we can be alone I'll be waiting, all there's left to do is run You'll be the prince and I'll be the princess It's a love story, baby, just say, Yes So I sneak out to the garden to see you We keep quiet, 'cause we're dead if they knew So close your eyes Escape this town for a little while, oh oh 'Cause you were Romeo, I was a scarlet letter And my daddy said, Stay away from Juliet But you were everything to me I was beggin' you, Please don't go,  and I said Romeo, take me somewhere we can be alone I'll be waiting, all there's left to do is run You'll be the prince and I'll be the princess It's a love story, baby, just say, Yes Romeo, save me, they're tryna tell me how to feel This love is difficult, but it's real Don't be afraid, we'll make it out of this mess It's a love story, baby, just say, Yes Oh, oh I got tired of waiting Wonderin' if you were ever comin' around My faith in you was fading When I met you on the outskirts of town, and I said Romeo, save me, I've been feeling so alone I keep waiting for you, but you never come Is this in my head? I don't know what to think He knelt to the ground and pulled out a ring And said, Marry me, Juliet You'll never have to be alone I love you and that's all I really know I talked to your dad, go pick out a white dress It's a love story, baby, just say, Yes Oh, oh, oh Oh, oh, oh, oh 'Cause we were both young when I first saw you"))
    # print('training')
    # print('evaluating')
    # print(nb.evaluate_classifier_accuracy(0.2))
