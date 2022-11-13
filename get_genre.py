# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
# import random
import nltk
# import joblib
import math
import pickle
from collections import Counter
# from scipy.sparse import lil_matrix
# from sklearn.linear_model import LogisticRegression as sklearn_LR
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
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
    """
    Return some representation of this text.
    At a minimum, you need to perform tokenization, the rest is up to you.
    """
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
    def __init__(self, train_data, test_data, tokenizer, rw1=None, rw2=None, rw3=None):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.tokenizer = tokenizer
        self.train_fn = train_data
        self.test_fn = test_data
        if rw2:
            with open('w2', 'rb') as w2f:
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
                                       "Jazz": 0.0
                                       }
        if rw3:
            with open('w3', 'rb') as w3f:
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
                                        "Jazz": 0.0
                                       }
        if rw1:
            with open('w1', 'rb') as w1f:
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
                                        "Jazz": Counter()
                                        }

    def train_model(self):
        with open(self.train_fn, encoding='utf-8') as csvfile:
            csvf = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvf:
                new_label = get_label(row[1:-1])
                if new_label == "OOV":
                    continue
                text = row[-1]
                self.tokenize_and_update_model(text, new_label)

        w1 = self.class_word_counts
        w2 = self.class_total_doc_counts
        w3 = self.class_total_word_counts

        with open('w1', 'wb') as w1_file:
            pickle.dump(w1, w1_file)
        with open('w2', 'wb') as w2_file:
            pickle.dump(w2, w2_file)
        with open('w3', 'wb') as w3_file:
            pickle.dump(w3, w3_file)

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
        return (self.class_word_counts[label].get(word, 0) + alpha) / (self.class_total_word_counts[label] + alpha * len(self.vocab))

    def log_likelihood(self, bow, label, alpha):
        return sum(math.log(self.p_word_given_label_and_alpha(word, label,alpha)) for word in bow.keys())

    def log_prior(self, label):
        label_docs = self.class_total_doc_counts[label]
        total_docs = sum(self.class_total_doc_counts.values())
        return math.log(label_docs / total_docs)

    def likelihood_ratio(self, word, alpha):
        return self.p_word_given_label_and_alpha(word, 'pos', alpha) / self.p_word_given_label_and_alpha(word, 'neg', alpha)

    def unnormalized_log_posterior(self, bow, label, alpha):
        return self.log_likelihood(bow, label, alpha) + self.log_prior(label)

    def classify(self, bow, alpha):
        labels = self.class_word_counts.keys()
        return sorted([(label, self.unnormalized_log_posterior(bow, label, alpha)) for label in labels], key= lambda x: -x[1])[0][0]

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
    # feat_name = classifier.featurizer
    print('acc {}'.format(train_acc*100))


def bag_of_words(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = nltk.word_tokenize(text.lower())
    tokens = list(filter(lambda token: token not in stop_words, tokens))
    feats = dict((token, 1) for token in tokens)

    return feats


nb = NaiveBayes(train_data='train_data.csv', test_data='test_data.csv', tokenizer=tokenize_doc_and_more, rw1='w1', rw2='w2', rw3='w3')
# nb.train_model()
# print(nb.evaluate_classifier_accuracy(0.2))


def classify_from_web(text):
    bow = nb.tokenizer(text)
    return nb.classify(bow, 0.2)


if __name__ == "__main__":
    artist_to_genre = dict()
    lyrics_to_artist = dict()
    lyrics_to_genre = dict()
    genres = set()


    # print('training')


    # print('evaluating')
    # print(nb.evaluate_classifier_accuracy(0.2))










    # bow_classifier = LogReg(bag_of_words)
    # bow_classifier.train_model("train_data.csv")
    # eval_class(bow_classifier)





























"""
class LogReg:

    def __init__(self, feat_method, min_feat=1, inv_reg_val=1.0):
        self.feat_idx = {}
        self.feat_method = feat_method
        self.min_feat = min_feat
        self.inv_reg_val = inv_reg_val

        self.model = None
        self.genres = set()

    def load_data(self, filename):
        bag_of_feats = []
        labels = []
        with open(filename, encoding='utf-8') as csvfile:
            csvf = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvf:
                llabels = row[1:-1]
                text = row[-1]
                labels.append(llabels)
                bag_of_feats.append(self.feat_method(text))
        return bag_of_feats, labels

    def build_feature_index(self, train_data):
        # if self.feat_idx:
        #    raise Exception('Feat index exists')
        #    exit(1)
        feature_dfs = Counter()
        for bag_of_feats in train_data:
            feature_dfs.update(bag_of_feats.keys())

        feat_id = 0
        for feat in feature_dfs:
            if feature_dfs[feat] >= self.min_feat:
                self.feat_idx[feat] = feat_id
                feat_id += 1

    def proc_data(self, filename, isTraining=False):
        bag_of_feats, labels = self.load_data(filename)

        if isTraining:
            self.build_feature_index(bag_of_feats)

        assert self.feat_idx, 'no feats'

        feat_vector_len = len(self.feat_idx)
        bag_of_feats_len = len(bag_of_feats)
        matrix = lil_matrix((bag_of_feats_len, feat_vector_len))
        for idx, bag_of_feats in enumerate(bag_of_feats):
            for feat, value in bag_of_feats.items():
                if feat in self.feat_idx:
                    feat_idx = self.feat_idx[feat]
                    matrix[idx, feat_idx] = value
        return matrix, labels

    def train_model(self, file):
        matrix, labels = self.proc_data(file, isTraining=True)
        nl = [l[0] for l in labels]
        # mlb = MultiLabelBinarizer()
        # arg = mlb.fit_transform(labels)
        # svm_model_linear = SVC(kernel='linear', C=1).fit(matrix, )

        self.model = sklearn_LR(penalty='l2', max_iter=5000, solver='liblinear', C=self.inv_reg_val, verbose=1)
        print('training')
        self.model.fit(matrix, nl)
        joblib.dump(self.model, "model.pkl")

    def evaluate_classifier(self, filename):
        assert self.model, 'No model'
        matrix, label = self.proc_data(filename)
        acc = self.model.score(matrix, label)
        return acc

    def predict(self, filename):
        assert self.model, "no model"
        matrix, labels = self.proc_data(filename)
        preds = self.model.predict(matrix)
        return preds
"""