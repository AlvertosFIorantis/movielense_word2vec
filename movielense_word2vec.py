
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import copy
import math
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim.models.word2vec as w2v
import multiprocessing
from datetime import datetime
import random
import os
import sys
sys.path.append(os.path.abspath('..'))
import json
from sklearn.utils import shuffle
import operator
import collections
from sklearn.metrics import mean_squared_error
import heapq
import pickle
r_cols = ['user_id', 'movie_id', 'rating',"time"]

ratings = pd.read_csv("./ml-100k/u1.base",
sep='\t', names=r_cols, usecols=range(4), encoding="ISO-8859-1")


m_cols = ['movie_id', 'title']

movies = pd.read_csv("./ml-100k/u.item",
sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

r_cols = ['user_id', 'movie_id', 'rating',"time"]

test_set = pd.read_csv("./ml-100k/u1.test",
sep='\t', names=r_cols, usecols=range(4), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)
test_set=pd.merge(movies, test_set)


'''chaninge unix time to normal time in order to be easier to understand'''
ratings['time'] = pd.to_datetime(ratings['time'],unit='s')
test_set['time'] = pd.to_datetime(test_set['time'],unit='s')


'''make all titles lower cases it might be needed letter'''

ratings["title"] = ratings["title"].map(lambda x: x if type(x)!=str else x.lower())

ratings.fillna(0)
test_set["title"] = test_set["title"].map(lambda x: x if type(x)!=str else x.lower())


userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
#print(userRatings.head())
users=list(userRatings.index.values)
movies=list(userRatings.columns.values)
matrix=userRatings.values
where_are_NaNs = np.isnan(matrix)
matrix[where_are_NaNs] = 0

'''make columns of the dataframe to lists for users, ratings, items'''
user_id=ratings.user_id.tolist()
item_id=ratings.movie_id.tolist()
labels=ratings.rating.tolist()
timestamp=ratings.time.tolist()
title=ratings.title.tolist()

'''make columns of the dataframe to lists for users ratings items for the test data set'''
user_id_test=test_set.user_id.tolist()
item_id_test=test_set.movie_id.tolist()
labels_test=test_set.rating.tolist()
timestamp_test=test_set.time.tolist()
title_test=test_set.title.tolist()

''' make a dictionary with key being a user id and value the siquence of movies based on the
rating each user have provied for each movie. For example if the rating is 4 for a specific
movie then the list have 4 times the item_id ( movie)'''

from collections import defaultdict
sentences= defaultdict(list)
count=0
for i in user_id:
    for j in range(labels[count]):
        lista=[]
        lista.append(title[count])
        sentences[i].extend(lista)
    count+=1
print("lenength of sentences",len(sentences))
'''creating a corpus based on the dictionary, with each nested list being a sentence with tokenized wordss'''
corpus=[]
for i in sentences:
    corpus.append(sentences[i])
print("length of corpus",len(corpus))
#print("position 1 of corpus",corpus[1])
sentences = copy.deepcopy(corpus)
print("length of sentences",len(sentences))
indexed_sentences = []
i = 0
word2idx = {}
for sentence in sentences:
    for token in sentence:
        token = token.lower()
        if token not in word2idx:
            word2idx[token] = i
            i += 1
for sentence in sentences:
    indexed_sentence = []
    for token in sentence:
        indexed_sentence.append(word2idx[token])
    indexed_sentences.append(indexed_sentence)


print ("Vocab size:", i)
print("length of indexed sentences",len(indexed_sentences))
#print("word2index",word2idx)
print("length of word2index",len(word2idx))


np.random.seed(10)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_weights(shape):
    return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))
class Model(object):
    def __init__(self, D, V, context_sz):
        self.D = D # embedding dimension
        self.V = V # vocab size
        #: we will look context_sz to the right AND context_sz to the left
        #so the total number of targets is 2*context_sz
        self.context_sz = context_sz
    def _get_pnw(self, X):
        # calculate Pn(w) - probability distribution for negative sampling
        # basically just the word probability ^ 3/4
        word_freq = {}
        word_count = sum(len(x) for x in X)
        for x in X:
            for xj in x:
                if xj not in word_freq:
                    word_freq[xj] = 0
                word_freq[xj] += 1
        self.Pnw = np.zeros(self.V)
        for j in range(self.V): # 0 and 1 are the start and end tokens, we won't use those here
            self.Pnw[j] = (word_freq[j] / float(word_count))**0.75
        # print "self.Pnw[2000]:", self.Pnw[2000]
        assert(np.all(self.Pnw[0:] > 0))
        return self.Pnw
    def _get_negative_samples(self, context, num_neg_samples):
        # temporarily save context values because we don't want to negative sample these
        saved = {}
        for context_idx in context:
            saved[context_idx] = self.Pnw[context_idx]
            # print "saving -- context id:", context_idx, "value:", self.Pnw[context_idx]
            self.Pnw[context_idx] = 0
        neg_samples = np.random.choice(
            range(self.V),
            size=num_neg_samples, # this is arbitrary - number of negative samples to take
            replace=False,
            p=self.Pnw / np.sum(self.Pnw),
        )
        # print "saved:", saved
        for j, pnwj in saved.items():
            self.Pnw[j] = pnwj
        assert(np.all(self.Pnw[0:] > 0))
        return neg_samples
    def fit(self, X, num_neg_samples=10, learning_rate=1e-4, mu=0.99, reg=0.1, epochs=10):
        N = len(X)
        V = self.V
        D = self.D
        self._get_pnw(X)
        # initialize weights and momentum changes
        self.W1 = init_weights((V, D))
        self.W2 = init_weights((D, V))
        dW1 = np.zeros(self.W1.shape)
        dW2 = np.zeros(self.W2.shape)
        costs = []
        cost_per_epoch = []
        sample_indices = range(N)
        for i in range(epochs):
            t0 = datetime.now()
            sample_indices = shuffle(sample_indices)
            cost_per_epoch_i = []
            for it in range(N):
                j = sample_indices[it]
                x = X[j] # one sentence
                # too short to do 1 iteration, skip
                if len(x) < 2 * self.context_sz + 1:
                    continue
                cj = []
                n = len(x)
                # for jj in xrange(n):
                ########## try one random window per sentence ###########
                jj = np.random.choice(n)
                # do the updates manually
                Z = self.W1[x[jj],:] # note: paper uses linear activation function
                start = max(0, jj - self.context_sz)
                end = min(n, jj + 1 + self.context_sz)
                context = np.concatenate([x[start:jj], x[(jj+1):end]])
                #context can contain DUPLICATES!
                # e.g. "<UNKOWN> <UNKOWN> cats and dogs"
                context = np.array(list(set(context)), dtype=np.int32)
                # print "context:", context
                posA = Z.dot(self.W2[:,context])
                pos_pY = sigmoid(posA)
                neg_samples = self._get_negative_samples(context, num_neg_samples)
                # technically can remove this line now but leave for sanity checking
                # neg_samples = np.setdiff1d(neg_samples, Y[j])
                # print "number of negative samples:", len(neg_samples)
                negA = Z.dot(self.W2[:,neg_samples])
                neg_pY = sigmoid(-negA)
                c = -np.log(pos_pY).sum() - np.log(neg_pY).sum()
                cj.append(c / (num_neg_samples + len(context)))
                # positive samples
                pos_err = pos_pY - 1
                dW2[:, context] = mu*dW2[:, context] - learning_rate*(np.outer(Z, pos_err) + reg*self.W2[:, context])
                # negative samples
                neg_err = 1 - neg_pY
                dW2[:, neg_samples] = mu*dW2[:, neg_samples] - learning_rate*(np.outer(Z, neg_err) + reg*self.W2[:, neg_samples])
                self.W2[:, context] += dW2[:, context]
                # self.W2[:, context] /= np.linalg.norm(self.W2[:, context], axis=1, keepdims=True)
                self.W2[:, neg_samples] += dW2[:, neg_samples]
                # self.W2[:, neg_samples] /= np.linalg.norm(self.W2[:, neg_samples], axis=1, keepdims=True)
                # input weights
                gradW1 = pos_err.dot(self.W2[:, context].T) + neg_err.dot(self.W2[:, neg_samples].T)
                dW1[x[jj], :] = mu*dW1[x[jj], :] - learning_rate*(gradW1 + reg*self.W1[x[jj], :])
                self.W1[x[jj], :] += dW1[x[jj], :]
                # self.W1[x[jj], :] /= np.linalg.norm(self.W1[x[jj], :])
                cj = np.mean(cj)
                cost_per_epoch_i.append(cj)
                costs.append(cj)
                if it % 500 == 0:
                    sys.stdout.write("epoch: %d j: %d/ %d cost: %f\r" % (i, it, N, cj))
                    sys.stdout.flush()
            epoch_cost = np.mean(cost_per_epoch_i)
            cost_per_epoch.append(epoch_cost)
            print ("time to complete epoch %d:" % i, (datetime.now() - t0), "cost:", epoch_cost)
        plt.plot(costs)
        plt.title("Error every 5000 words")
        plt.show()
        plt.plot(cost_per_epoch)
        plt.title("Word Embeddings cost at each epoch")
        plt.show()
    def save(self, fn):
        arrays = [self.W1, self.W2]
        np.savez(fn, *arrays)
    def runmodel():
        sentences=indexed_sentences

        V = len(word2idx)
        model = Model(10, V, 65)
        model.fit(sentences, learning_rate=0.025, mu=0.005, epochs=5, num_neg_samples=5)
        model.save('w2v_model.npz')
    def wordembedding(we_file='w2v_model.npz'):
        npz = np.load(we_file)
        W1 = npz['arr_0']
        W2 = npz['arr_1']
        We = (W1+ W2.T)/2
        return(We)
a=Model.runmodel()
Q=Model.wordembedding()
print(len(Q))
print(Q[word2idx["toy story (1995)"]])
print("Star Wars",Q[word2idx["star wars (1977)"]])
print("empire strikes back, the (1980)",Q[word2idx["empire strikes back, the (1980)"]])
print("Return of the Jedi (1983)",Q[word2idx["return of the jedi (1983)"]])
print("four rooms (1995)",Q[word2idx["four rooms (1995)"]])
print("babe (1995)",Q[word2idx["babe (1995)"]])
text=["star wars (1977)","empire strikes back, the (1980)","return of the jedi (1983)","babe (1995)"]

'''TSNE'''
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = copy.deepcopy(Q)
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
points=pd.DataFrame(data=all_word_vectors_matrix_2d,columns=["x", "y"],index=word2idx)
print(points.head(5))
'''ax=points[30:31].plot.scatter("x", "y", s=10, figsize=(20, 12))
for i, txt in enumerate(points.index[30:31]):
    ax.annotate(txt, (points.x[i],points.y[i]))
plt.show()'''
fig, ax = plt.subplots()
points[220:230].plot('x', 'y', kind='scatter', ax=ax)

for k, v in points[220:230].iterrows():
    ax.annotate(k, v)
plt.show()





'''P intial matrix for (N,K) N being number of user K = number of features'''

num_features=10
P=np.random.rand(len(users),num_features)
user_bias = np.zeros(len(users))
item_bias = np.zeros(len(movies))

def matrix_factorization(matrix,P, Q, steps=100, alpha=0.001,beta=0.01,item_bias_reg=0.01,user_bias_reg=0.01):
    Q = Q.T
    user_bias = np.zeros(len(users))
    item_bias = np.zeros(len(movies))
    global_bias = np.mean(matrix[np.where(matrix != 0)])
    for step in range(steps):
        if step%10==0:
            print(step)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] > 0:
                    eij = matrix[i][j] - (np.dot(P[i,:],Q[:,j])+user_bias[i]+item_bias[j]+global_bias)

                    user_bias[i] += alpha * (eij -beta * user_bias[i])
                    item_bias[j] += alpha * (eij - beta * item_bias[j])
                    for k in range(num_features):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])


        eR = np.dot(P,Q)
        e = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if matrix[i][j] > 0:
                    e = e + pow(matrix[i][j] - (np.dot(P[i,:],Q[:,j])+user_bias[i]+item_bias[j]+global_bias), 2)
                    e = e + (beta) *  pow(user_bias[i],2)
                    e = e + (beta) *  pow(item_bias[j],2)
                    for k in range(num_features):
                        e = e + (beta) *  pow(P[i][k],2)
        if e < 0.001:
            break

    np.save('user_bias_error.npy', user_bias)
    np.save('item_bias_error.npy', item_bias)
    return P

if os.path.exists("biases_MF_error.pickle"):
    model_f=open("biases_MF_error.pickle","rb")
    model=pickle.load(model_f)
    model_f.close()
    user_bias = np.load('user_bias_error.npy')
    item_bias = np.load('item_bias_error.npy')
else:
    model=matrix_factorization(matrix,P,Q)
    save_model=open("biases_MF_error.pickle","wb")
    pickle.dump(model,save_model)
    save_model.close()
    user_bias = np.load('user_bias_error.npy')
    item_bias = np.load('item_bias_error.npy')


mode_matrix_factorization=np.dot(model,Q.T)
global_bias = np.mean(matrix[np.where(matrix != 0)])



def predict_ranking_matrix( user, movie):
    uidx = users.index(user)
    midx = movies.index(movie)
    global_bias = np.mean(matrix[np.where(matrix != 0)])
    predicted_rating=user_bias[uidx]+item_bias[midx]+global_bias

    if matrix[uidx, midx] > 0:
        return None
    else:
        predicted_rating+=mode_matrix_factorization[uidx, midx]
        return predicted_rating

predicted_scores_matrix=[]
clean_user_id_test=[]
clean_title_test=[]
clean_labels_test=[]

for i in range(len(user_id_test)):
    if title_test[i]  in title:
        clean_user_id_test.append(user_id_test[i])
        clean_title_test.append(title_test[i])
        clean_labels_test.append(labels_test[i])


print(predict_ranking_matrix(clean_user_id_test[120],clean_title_test[120]))


for i in range(len(clean_user_id_test)):
    a=predict_ranking_matrix(clean_user_id_test[i],clean_title_test[i])
    predicted_scores_matrix.append(a)
print(len(predicted_scores_matrix))
clean_matrix_pred=[]
clean_matrix_test=[]
for i in range(len(predicted_scores_matrix)):
    if predicted_scores_matrix[i]!=None:
        clean_matrix_pred.append(predicted_scores_matrix[i])
        clean_matrix_test.append(clean_labels_test[i])

count=[]
for i in range(len(predicted_scores_matrix)):
    if predicted_scores_matrix[i]==None:
        count.append(i)

for i in count:
    if clean_title_test[i]  in word2idx:
        print(clean_title_test[i], "word2idx")

rms_matrix = math.sqrt(mean_squared_error(clean_matrix_test, clean_matrix_pred))
print("RMSE IS ",rms_matrix)


