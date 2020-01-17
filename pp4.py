from scipy.special import expit
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import random
import copy
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


data_folder = "pp4data/20newsgroups/"
set_of_all_words = set()
unique_words = []
duplicate_words = []
w_n = []
z_n = []
d_n = []
K = 20
alpha = 5 / K
beta = 0.01
n_iters = 500


# Function to split data in train test set
def train_test_split(data_file, label_file):
    n = len(data_file)
    index = np.arange(n)
    random.shuffle(index)
    # Split file in train,test data with test data as one third of total data
    test_index = index[:int(n / 3)]
    test_data = data_file[test_index]
    test_label = label_file[test_index]
    train_index = index[int(n / 3):]
    train_data = data_file[(train_index[:int(len(train_index))])]
    train_label = label_file[(train_index[:int(len(train_index))])]
    return train_data, test_data, train_label, test_label

# Function to create list of words both duplicates and unique
def word_doc(filename):
    with open(data_folder + filename, "r") as f:
        input_data = []
        for line in f.read().split("\n"):
            for word in line.strip().split():
                input_data.append(word)
                duplicate_words.append(word)
                set_of_all_words.add(word)
    return input_data


# Function to read each document in the folder and to create dictionary of words and labels
def read_data(filename):
    dict_of_words = dict()
    label_dict = dict()
    data = np.genfromtxt(filename, delimiter=',', dtype=int)
    for i in range(len(data)):
        dict_of_words[data[i][0]] = word_doc(str(data[i][0]))
        label_dict[data[i][0]] = data[i][1]
    return dict_of_words, label_dict


# Function for collapsed Gibbs Sampling
# All the variables are as per the algorithm given in the pp4.pdf
def gibbs_sampling(dict_of_words):
    N_docs = len(dict_of_words)
    N_words = len(duplicate_words)
    cd = np.zeros((N_docs, K))
    ct = np.zeros((K, len(unique_words)))
    p = [0] * K

    pi_n = list(range(0, len(duplicate_words)))
    random.shuffle(pi_n)

    for doc in dict_of_words:
        for word in dict_of_words[doc]:
            z_n.append(random.randint(0, K - 1))
            d_n.append(doc - 1)
            w_n.append(unique_words.index(word))

    for i in range(len(duplicate_words)):
        cd[d_n[i]][z_n[i]] += 1
        ct[z_n[i]][w_n[i]] += 1

    for i in range(0, n_iters):
        for n in range(0, N_words):
            word = w_n[pi_n[n]]
            topic = z_n[pi_n[n]]
            doc = d_n[pi_n[n]]
            cd[doc][topic] = cd[doc][topic] - 1
            ct[topic][word] = ct[topic][word] - 1

            for k in range(0, K):
                p[k] = (ct[k][word] + beta) * (cd[doc][k] + alpha) / (
                            (len(unique_words) * beta + np.sum(ct[k, :])) * (K * alpha + np.sum(cd[doc, :])))
            p = np.divide(p, np.sum(p))
            topic = np.random.choice(range(0, K), p=p)
            z_n[pi_n[n]] = topic
            cd[doc][topic] = cd[doc][topic] + 1
            ct[topic][word] = ct[topic][word] + 1

    return z_n, cd, ct


# Function to calculate sigmoid
# Expit is a direct and fast function to calculate sigmoid values
def sigmoid(x):
    return expit(x)


# glm bayesian function for logistic model
def glm_logistic(phi, t, phi_test, t_test, alpha):
    phi = np.asarray(phi)
    w = np.zeros((phi.shape[1], 1))
    w_old = w
    n = 0
    while n < 100:
        yi = expit(np.dot(phi, w_old))
        d = np.reshape(t, (len(t), 1)) - yi
        r = yi * (1 - yi)
        R = np.diagflat(r)
        fd = np.dot(phi.T, d) - (alpha * w_old)
        sd = - (np.dot(phi.T, np.dot(R, phi))) - (alpha * (np.identity(phi.shape[1])))
        w_new = w_old - np.dot(np.linalg.inv(sd), fd)

        # Convergence condition of wmap
        converge = (np.linalg.norm(w_new - w_old, 2)) / np.linalg.norm(w_old, 2)
        if converge < pow(10, -3):
            w_old = w_new
            break
        w_old = w_new
        n += 1

    w_map = w_old
    error_count = 0

    t_predict = np.dot(np.array(phi_test), np.array(w_map))

    # Calculating error counts
    for i in range(t_predict.shape[0]):
        if t_predict[i] >= 0.5:
            t_predict[i] = 1
        else:
            t_predict[i] = 0
        if t_predict[i] != t_test[i]:
            error_count += 1

    return error_count / t_predict.shape[0]

accuracy = []
std_dev = []

# Calculate performance, mean and std deviation for plotting
def calculate_acc(phi, t, alpha):
    error = []
    for i in range(0, 30):
        phi_train, phi_test, t_train, t_test = train_test_split(phi, t)
        size = len(phi_train)
        error_count_list = []

        for point in range(10):
            phi_sample = phi_train[0:int(0.1 * size * (point + 1))]
            t_sample = t_train[0:int(0.1 * size * (point + 1))]
            error_count = glm_logistic(phi_sample, t_sample, phi_test, t_test, alpha)
            error_count_list.append(error_count)

        error.insert(i, error_count_list)

    # Calculating average error rate for each train set size
    mean_list = []
    for t_size in range(0, 10):
        sum = 0
        for iter in range(30):
            sum += error[iter][t_size]
        mean_list.insert(t_size, sum / 30)

    # Calculating standard deviation
    std_deviation_list = []
    for t_size in range(10):
        std = 0
        for iter in range(30):
            std += (error[iter][t_size] - mean_list[t_size]) ** 2
        std_deviation_list.append(math.sqrt(std / 30))

    acc = [1-x for x in mean_list]
    accuracy.append(acc)
    std_dev.append(std_deviation_list)


if __name__ == "__main__":
    print('Started............')
    dict_of_words, label_dict = read_data(data_folder + 'index.csv')
    unique_words = list(set_of_all_words)
    print('\nData Parsing done')
    z_n, cd, ct = gibbs_sampling(dict_of_words)
    print('\nGibbs sampling done')

    topic_word_dict = {topic: dict() for topic in range(ct.shape[0])}

    topics = {}
    for i in range(len(ct)):
        for j in ct[i].argsort()[-5:][::-1]:
            if i in topics:
                topics[i] += [unique_words[j]]
            else:
                topics[i] = [unique_words[j]]

    (pd.DataFrame.from_dict(data=topics, orient='index')
     .to_csv('topicwords.csv', header=False))

    X_LDA = copy.deepcopy(cd)
    alpha = 0.01
    for i in range(cd.shape[0]):
        for j in range(0, K):
            X_LDA[i][j] = X_LDA[i][j] + alpha / (
                        K * alpha + np.sum(X_LDA[i, :]))

    X_BAG = np.zeros((cd.shape[0], ct.shape[1]), dtype=int)

    for d in dict_of_words:
        for each in dict_of_words[d]:
            X_BAG[d - 1][unique_words.index(each)] += 1

    Y = list(label_dict.values())
    Y = np.array(Y)
    np.reshape(Y, (len(Y), 1))
    calculate_acc(X_LDA, Y, alpha=0.01)
    calculate_acc(X_BAG, Y, alpha=0.01)
    train_size = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    plt.errorbar(train_size, accuracy[0], std_dev[0], capsize=20, label="LDA", color='green')
    plt.errorbar(train_size, accuracy[1], std_dev[1], capsize=20, label="Bag of Words", color='red')
    plt.xlabel('Train Set Size')
    plt.ylabel('Accuracy')
    plt.title('LDA vs bag of words: Accuracy on 20newsgroup dataset')
    plt.legend()
    plt.savefig('LDAvsBoW.jpeg')

    print('Graph has been plotted and saved')