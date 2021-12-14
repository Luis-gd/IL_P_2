#Imports
import os
import nltk
import math
import numpy as np
import sklearn

from scipy.special import logsumexp
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
from sklearn.naive_bayes import GaussianNB
import il_script1__vs as z
import il_script2__tf_idf as y
import il_script3__nb as x
import il_script4__nn as w
import matplotlib.pyplot as plt


def plot_test(interval, model, max):
    ex_t  = []
    acc = []
    x_axis = []
    xx = []
    for i in range(max-1):
        top_k = interval*(i+1)
        x_axis.append(top_k)
        if model == 'nb':
            xx = x.run_nb(top_k, False, False)
            ex_t.append(xx[1])
            acc.append(xx[0])
        elif model == 'vs-2':
            xx = z.run_tf_idf(top_k, 'cosine', False, False)
            ex_t.append(xx[1])
            acc.append(xx[0])
        elif model == 'vs':
            xx = y.run_tf_idf(top_k, 'cosine', False, False)
            ex_t.append(xx[1])
            acc.append(xx[0])
        elif model == 'nn':
            xx = w.run_nn(top_k, False, False)
            ex_t.append(xx[1])
            acc.append(xx[0])

    plt.plot(x_axis,acc,'b')
    plt.show()
    plt.plot(x_axis, ex_t,'r')
    plt.show()
