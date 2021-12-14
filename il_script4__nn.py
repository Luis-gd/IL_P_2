#Imports
import os
import nltk
import math
import numpy as np
import time
import pandas as pd
import tensorflow as tf


from scipy.special import logsumexp
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
from tensorflow import keras

###############################################################################
################### PRE PROCESSING ############################################
###############################################################################
#return [train, test]
def read(categories):
    #Import stop words:
    #nltk.download('stopwords')
    #nltk.download('punkt')
    stop_words = set(stopwords.words('spanish'))
    #print(stopwords.words())

    #Tokenizer
    tokenizer = RegexpTokenizer(r'[A-Za-z_À-ÿ]{3,}')
    #Read files, tokenize and remove stopwords
    train = []
    test = []


    j=0#j=0->deportes, j=1->salud, j=2->politica

    for i in range(30):
        j=0#j=0->deportes, j=1->salud, j=2->politica
        for cat in categories:
            #Train set
            path = os.path.dirname(__file__)+"/dataset/"+cat+"/"+cat+str(2*i+1)+".txt"
            with open(path) as f:
                text = f.read()
                text_tokens = tokenizer.tokenize(text)
                filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
                train.append([filtered_text,j])
            #Test set
            path = os.path.dirname(__file__)+"/dataset/"+cat+"/"+cat+str(2*i+2)+".txt"
            with open(path) as f:
                 text = f.read()
                 text_tokens = tokenizer.tokenize(text)
                 filtered_text = [w for w in text_tokens if not w.lower() in stop_words]
                 test.append([filtered_text,j])

            j+=1

    return [train, test]


###############################################################################
####################### TF-IDF ################################################
###############################################################################

#Compute IDF glossary of the entire training set
#return [IDF, union]
def idf(train, test):
    union = []
    N = len(train)+len(test)
    #Union set of all tokens
    for doc in train:
        union = set(union).union(set(doc[0]))

    #union_train = union

    for doc in test:
        union = set(union).union(set(doc[0]))

    #IDF  initialization

    IDF = dict.fromkeys(union,0)

    #Compute IDF values
    for token in union:
        #Train docs
        for doc in train:
            if token in doc[0]:
                IDF[token] += 1
        #Test document
        for doc in test:
            if token in doc[0]:
                IDF[token] += 1

    #IDF Values
    for token in union:
        IDF[token] = math.log10((N/(1+IDF[token])))


    return [IDF, union]

#Compute tf-idf glossary of a single document with label [tokenized_text, label]
#Computes array and glossary with top_k terms
# return [arr, l]
def tf_idf(doc, idf, top_k, categories, union):
    dic_doc_tf = dict.fromkeys(union,0)
    dic_doc_tf_idf = dict.fromkeys(union,0)
    #TF
    all_zero_tf = True
    for token in doc[0]:
        dic_doc_tf[token] += 1
        if dic_doc_tf[token] > 0:
            all_zero_tf = False

    if(all_zero_tf):
        print("# WARNING: All zero tf")
        print("Category "+categories[doc[1]]+" Doc:\n"+str(doc[0]))

    for token in doc[0]:
        dic_doc_tf_idf[token] = (idf[token]) * (dic_doc_tf[token])

    l = sorted(dic_doc_tf_idf.items(), key=itemgetter(1), reverse=True)
    l = l[:top_k]
    #Compute array
    arr = []
    k=0
    for token in union:
        match = False
        for tuple in l:
            if token == tuple[0]:
                match = True
                break
        if match:
            arr.append(1)
            k += 1
        else:
            arr.append(0)

    if k < top_k:
        print("WARNING: j = "+str(j))
    return [arr, l]

# Compute tf-idf Glosary for elements of train or test:
#return X, y
def glossary_create(train, IDF, top_k, categories,union):
    X = []
    Y = []
    for doc in train:
        tuple = [0]*len(categories)
        l = tf_idf(doc,IDF,top_k,categories,union)
        X.append(l[0])
        tuple[doc[1]]=1
        Y.append(tuple)

    return [X, Y];

########################## NAÏVE BAYES ########################################

def train_nn(X, Y, XTest, YTest, verb):
    #hyperparameters
    n_epochs = 1000
    learning_rate = 0.1
    n_neurons_per_hlayer = [100]
    dropout_rate = 0.01
    l2reg = 0.01
    batch_size=len(X)
    layer_activation = "elu"
    INPUTS = len(X[0])
    OUTPUTS = len(Y[0])
    model = keras.Sequential(name="DeepFeedforward")
    model.add(keras.layers.InputLayer(input_shape=(INPUTS,), batch_size=None))
    my_initializer = keras.initializers.he_uniform(seed=None)
    my_regularizer = keras.regularizers.l2(l2reg);
    for neurons in n_neurons_per_hlayer:
      model.add(keras.layers.Dense(neurons, activation=layer_activation, kernel_initializer = my_initializer,kernel_regularizer=my_regularizer))
      model.add(keras.layers.BatchNormalization())
      model.add(keras.layers.Dropout(rate=dropout_rate))

    model.add(keras.layers.Dense(OUTPUTS, activation="softmax"))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
              metrics=["categorical_accuracy"])

    history = model.fit(X, Y, batch_size=batch_size, epochs=n_epochs, verbose=0, validation_data=(XTest, YTest))
    results=pd.DataFrame(history.history)
    if verb:
        print(results[-1:])
    return results.val_categorical_accuracy.values[-1:][0]

def run_nn(top_k, verbose, verbose2):
    t0 = time.time()
    categories = ["deportes", "salud", "politica"]
    data = read(categories);
    idf_union = idf(data[0],data[1])
    train_set_vectors_tuples = glossary_create(data[0], idf_union[0],top_k, categories, idf_union[1])
    test_set_vectors_tuples = glossary_create(data[1], idf_union[0],top_k, categories, idf_union[1])
    acc = train_nn(train_set_vectors_tuples[0], train_set_vectors_tuples[1], test_set_vectors_tuples[0], test_set_vectors_tuples[1], verbose)
    t1 = time.time()
    print("Accuracy: "+str(acc))
    return [acc, (t1-t0)]
