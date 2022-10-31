# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Interface.ui',
# licensing of 'Interface.ui' applies.
#
# Created: Sun Dec  1 16:41:04 2019
#      by: pyside2-uic  running on PySide2 5.13.2
#
# WARNING! All changes made in this file will be lost!
# Qt Designer에서 만든 ui파일을 py파일로 변환하였음

import sys
import numpy as np
from scipy.sparse import csr_matrix
from PySide2 import QtCore, QtWidgets
from sklearn.metrics import pairwise_distances
from krwordrank.word import KRWordRank
from soynlp.tokenizer import MaxScoreTokenizer

class KeywordVectorizer:
    def __init__(self, tokenize, vocab_score):
        self.tokenize = tokenize
        self.idx_to_vocab = [vocab for vocab in sorted(vocab_score, key=lambda x:-vocab_score[x])]
        self.vocab_to_idx = {vocab:idx for idx, vocab in enumerate(self.idx_to_vocab)}
        self.keyword_vector = np.asarray([score for _, score in sorted(vocab_score.items(), key=lambda x:-x[1])])
        self.keyword_vector = self._L2_normalize(self.keyword_vector)

    def _L2_normalize(self, vectors):
        return vectors / np.sqrt((vectors ** 2).sum())

    def vectorize(self, sents):
        rows, cols, data = [], [], []
        for i, sent in enumerate(sents):
            terms = set(self.tokenize(sent))
            for term in terms:
                j = self.vocab_to_idx.get(term, -1)
                if j == -1:
                    continue
                rows.append(i)
                cols.append(j)
                data.append(1)
        n_docs = len(sents)
        n_terms = len(self.idx_to_vocab)
        return csr_matrix((data, (rows, cols)), shape = (n_docs, n_terms))

def summarize_with_sentences(texts, num_keywords = 100, num_keysents = 10, diversity = 0.3, stopwords = None, scaling = None, penalty = None, min_count = 5, max_length = 10, beta = 0.85, max_iter = 10, num_rset = -1, verbose = False):
    wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length, verbose=verbose)
    num_keywords_ = num_keywords
    if stopwords is not None:
        num_keywords_ += len(stopwords)
    keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter, num_keywords=num_keywords_, num_rset=num_rset)
    if scaling is None:
        scaling = lambda x: np.sqrt(x)
    if stopwords is None:
        stopwords = {}
    vocab_score = make_vocab_score(keywords, stopwords, scaling=scaling, topk=num_keywords)
    tokenizer = MaxScoreTokenizer(scores=vocab_score)
    sents = keysentence(vocab_score, texts, tokenizer.tokenize, num_keysents, diversity, penalty)
    keywords_ = {vocab: keywords[vocab] for vocab in vocab_score}
    return keywords_, sents

def keysentence(vocab_score, texts, tokenize, topk = 10, diversity = 0.3, penalty = None):
    if not callable(penalty):
        penalty = lambda x: 0
    if not 0 <= diversity <= 1:
        raise ValueError('Diversity must be [0, 1] float value')

    vectorizer = KeywordVectorizer(tokenize, vocab_score)
    x = vectorizer.vectorize(texts)
    keyvec = vectorizer.keyword_vector.reshape(1, -1)
    initial_penalty = np.asarray([penalty(sent) for sent in texts])
    idxs = select(x, keyvec, texts, initial_penalty, topk, diversity)
    return [texts[idx] for idx in idxs]

def select(x, keyvec, texts, initial_penalty, topk = 10, diversity = 0.3):
    dist = pairwise_distances(x, keyvec, metric = 'cosine').reshape(-1)
    dist = dist + initial_penalty
    idxs = []
    for _ in range(topk):
        idx = dist.argmin()
        idxs.append(idx)
        dist[idx] += 2
        idx_all_distance = pairwise_distances(x, x[idx].reshape(1, -1), metric = 'cosine').reshape(-1)
        penalty = np.zeros(idx_all_distance.shape[0])
        penalty[np.where(idx_all_distance < 0.7)[0]] = 2
        dist += penalty
    return idxs

def make_vocab_score(keywords, stopwords, negatives = None, scaling = lambda x:x, topk = 100):
    if negatives is None:
        negatives = {}
    keywords_ = {}
    for word, rank in sorted(keywords.items(), key = lambda x:-x[1]):
        if len(keywords_) >= topk:
            break
        if word in stopwords:
            continue
        if word in negatives:
            keywords_[word] = negatives[word]
        else:
            keywords_[word] = scaling(rank)

        return keywords_

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(691, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 201, 16))
        self.label.setObjectName("label")
        self.outputbox = QtWidgets.QTextBrowser(self.centralwidget)
        self.outputbox.setGeometry(QtCore.QRect(10, 290, 561, 271))
        self.outputbox.setObjectName("outputbox")
        self.inputbox = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.inputbox.setGeometry(QtCore.QRect(10, 40, 561, 191))
        self.inputbox.setObjectName("inputbox")
        self.analizebutton = QtWidgets.QPushButton(self.centralwidget)
        self.analizebutton.setGeometry(QtCore.QRect(580, 200, 93, 28))
        self.analizebutton.setObjectName("analizebutton")
        self.refreshbutton = QtWidgets.QPushButton(self.centralwidget)
        self.refreshbutton.setGeometry(QtCore.QRect(580, 530, 93, 28))
        self.refreshbutton.setObjectName("refreshbutton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 260, 191, 16))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "MainWindow", None, -1))
        self.label.setText(QtWidgets.QApplication.translate("MainWindow", "분석할 텍스트를 입력하세요.", None, -1))
        self.analizebutton.setText(QtWidgets.QApplication.translate("MainWindow", "분석하기", None, -1))
        self.refreshbutton.setText(QtWidgets.QApplication.translate("MainWindow", "새로고침", None, -1))
        self.label_2.setText(QtWidgets.QApplication.translate("MainWindow", "분석 결과 입니다.", None, -1))

        self.refreshbutton.clicked.connect(self.del_refresh)
        self.analizebutton.clicked.connect(self.Analize)

    def del_refresh(self):
        self.inputbox.clear()
        self.outputbox.clear()

    def Analize(self):
        RawData = self.inputbox.toPlainText()
        Data = RawData.split(".")
        penalty = lambda x:0 if (25 <= len(x) <= 150) else 1
        keywords, sents = summarize_with_sentences(Data, penalty = penalty, diversity = 0.3, num_keywords = 100, num_keysents = 10, scaling = None, verbose = False)
        for i in sents:
            self.outputbox.append(i)

def highlight_keyword(sent, keywords):
    for keyword, score in keywords.items():
        if score > 0:
            sent = sent.replace(keyword, '[&s]' % keyword)
    return sent

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

