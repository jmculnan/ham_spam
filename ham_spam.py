
####################IMPORTING##########################################
###import necessary modules
from math import exp, floor
from collections import Counter
import numpy as np
import re
import pickle
np.set_printoptions(threshold=np.inf)

#######################################################################
###shorten the file handles for ease
trainset = 'SMSSpamCollection.train'
devset   = 'SMSSpamCollection.devel'
testset  = 'SMSSpamCollection.test'

####################DEFINE SMALL FUNCTIONS#############################
###1.TOKENIZE UNIGRAMS
def uni_tokenize(sentenceset):
    tokenfile = []
    with open(sentenceset, 'r') as spamfile:
        for line in spamfile:
            tokenfile.append(re.findall(r"[\w']+|[.,!?;]", line.strip('\n')))
    return tokenfile

###1b. TOKENIZE UNIGRAMS + BIGRAMS
def bi_tokenize(sentenceset):
    total_tokens = []
    sentences    = []
    tokens       = uni_tokenize(sentenceset)
    grams_per_t  = []
    for phrase in tokens:
        sentences.append(' '.join(phrase[1:]))
    for sent in sentences:
        throwaway = []
        for gram in zip(sent.split(" ")[:-1], sent.split(" ")[1:]):
            throwaway.append(gram)
        grams_per_t.append(throwaway)
    for i in range(len(tokens)):
        junk = []
        junk.extend(tokens[i])
        junk.extend(grams_per_t[i])
        total_tokens.append(junk)
    return total_tokens

###2. CALCULATE LENGTH OF DATA IN LINES
def get_setsize(sentenceset):
    setsize = 0
    with open(sentenceset, 'r') as spamfile:
        for line in spamfile:
            setsize += 1
    return setsize

###3. CREATE LIST OF VOCABULARY
def create_vocab(dataset, uni_bi='unigram'):
    types = []
    if uni_bi == 'bigram':
        unigrams = []
        bigrams  = []
        tokens = bi_tokenize(dataset)
        for phrase in tokens:
            for word in phrase:
                if type(word) == tuple:
                    bigrams.append(word)
                elif type(word) == str:
                    unigrams.append(word)
        unigrams = list(set(unigrams))
        bigrams  = list(set(bigrams))
        unigrams.sort()
        bigrams.sort()
        types.extend(bigrams)
        types.extend(unigrams)
    else:
        tokens = uni_tokenize(dataset)
        for phrase in tokens:
            [types.append(word) for word in phrase[1:] if word not in types]
        types.sort()
    return types

###4. CREATE VECTORS FOR DATA IN EACH TEXT
def create_filled_vectors(dataset, uni_bi='unigram'):
    vectors = []
    if uni_bi == 'bigram':
        tokenfile = bi_tokenize(dataset)
    else:
        tokenfile = uni_tokenize(dataset)
    types = create_vocab(trainset, uni_bi)
    for phrase in tokenfile:
        output  = []
        if phrase[0] == 'ham':
            output.append(0)
        else:
            output.append(1)
        output.append(1)
        for item in types:
            if item in phrase[1:]:
                output.append(phrase.count(item))
            else:
                output.append(0)
        vectors.append(output)
    return vectors

###5. CREATE DICTIONARY OF FREQUENCIES
def vocab_freq(tokenset):
    totals = {}
    for phrase in tokenset:
        #calculate frequencies
        for word in phrase[1:]:
            if word not in totals:
                totals[word] = 1
            else:
                totals[word] += 1
    return totals

def sigmoid(vector, weight_vector):
    z = np.dot(vector, weight_vector)
    if z >= 0:
        h = 1 / (1 + exp(-z))
    else:
        h = exp(z) / (1 + exp(z))
    return h  
 
##define a function to train the data
def train(ep=10, mini=10, al=.1, uni_bi='unigram', pickle_name='trained_theta'):
    epochs    = int(ep)
    minibatch = int(mini)
    alpha     = float(al)
    count   = 0 #counts number of epochs 
    idx     = 0 #starting index to cycle through data
    ##create data structure
    vectors = create_filled_vectors(trainset, uni_bi)
    update  = np.zeros(len(vectors[0]) -1)
    while count < epochs:
        #counts number incorrect tags
        incorr_count = 0
        #shuffle the data
        np.random.shuffle(vectors)
        theta = np.zeros(len(vectors[0]) - 1) #one array of 8436 zeros
            ##calculates number of minibatches
        for q in range(floor(len(vectors) / minibatch)):
                #create data structure to save update values per minibatch
            update_part = np.zeros(len(vectors[0]) - 1)
            for i in range(minibatch):
                ##index keeps track of place in data
                idx = q * minibatch + i
                #vector for vocab + bias for each text
                x_sub_i = np.asarray(vectors[idx][1:])
                #tag value, 0 or 1, for each text
                y = vectors[idx][0]
                #calculates sigmoid
                h = sigmoid(theta, x_sub_i)
                update_part += alpha * x_sub_i * (y - h)
            update += update_part / minibatch
            theta += update  
        count += 1
        pickle.dump(theta, open(pickle_name, "wb"))           
    return theta

####4. TUNE THE DATA#####################################################
def examine(dataset, uni_bi='unigram', pickle_name='trained_theta'):
    #load saved model to ensure max accuracy
    tuning_theta = pickle.load( open(pickle_name, "rb"))
    #use this to keep track of the size of the devset
    setsize      = 0
    idx          = 0 #starting index to cycle through data
    incorr_count = 0
    vectors = create_filled_vectors(devset, uni_bi)
    for i in range(len(vectors)):
        setsize += 1
        x_sub_i = np.asarray(vectors[i][1:])
        y = vectors[i][0]        
        #calculates exponent
        h = sigmoid(tuning_theta, x_sub_i)
        if round(h) != y:
            incorr_count += 1
    accuracy = (setsize - incorr_count) / setsize           
    return accuracy

def tune(epochs, alphas, minibatches, uni_bi='unigram', pickle_name='trained_theta', filename='test.txt'):
    with open(filename, 'w') as savefile:
        trial_num = 0
        store_info     = []
        store_accuracy = []
        for epoch in epochs:
            for alpha in alphas:
                for minibatch in minibatches:
                    trial_num += 1
                    theta    = train(epoch, minibatch, alpha, uni_bi, pickle_name)
                    accuracy = examine('SMSSpamCollection.devel', uni_bi, pickle_name)
                    savefile.write('epochs: %s\t minibatch: %s\t alpha: %s\t accuracy: %s\n' \
                    % (str(epoch), str(minibatch), str(alpha), str(accuracy)))
                    store_info.append([epoch, minibatch, alpha])
                    store_accuracy.append(accuracy)
        best = [max(store_accuracy), store_info[store_accuracy.index(max(store_accuracy))]]
        return 'the best values are:\nepochs = \t%d\nminibatch = \t%d\nalpha = \t%s\n' \
         % (best[1][0], best[1][1], str(best[1][2])) + \
        'with an accuracy of %f percent on the development set.' % (best[0])

print(tune([5, 10, 20, 40, 50], [.01, .1, 1], [10, 15, 20], 'unigram', 'tuning_test'))

def test(pickle_name, uni_bi='unigram'):
    print(examine(pickle_name, uni_bi, testset))

def top_ham_features(featureset, features):
    feature_weights = pickle.load(open(featureset, "rb"))
    feats_w_weights = list(zip(features, feature_weights))
    sorted_feats = sorted(feats_w_weights, key=lambda x: x[1])
    return sorted_feats[:20]

def top_spam_features(featureset, features):
    feature_weights = pickle.load(open(featureset, "rb"))
    feats_w_weights = list(zip(features, feature_weights))
    sorted_feats = sorted(feats_w_weights, key=lambda x: x[1], reverse=True)
    return sorted_feats[:20]

def get_frequency(word):
    if word in totals.keys():
        return totals[word]
    else:
        return None
