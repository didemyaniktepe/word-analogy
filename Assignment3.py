import gensim.models.keyedvectors as word2vec
import numpy as np
from numpy.linalg import norm

################TASK1 Analogy Experiments################


model = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True,limit=10000)


def load_file(file_name):


    section = None
    first_line = 0
    all_word = set()
    all_sections = {}

    with open(file_name, 'r') as f:
        for line in f:
            if first_line == 0:
                first_line = first_line + 1
                continue
            elif line[0] == (':'):
                section = line.strip()[2:]
                continue
            try:
                all_sections[section].append(line.strip().split())
            except(KeyError):
                all_sections[section] = []
                all_sections[section].append(line.strip().split())
            words = line.strip().split()
            for word in words:
                all_word.add(word)


    needed_list = ["capital-common-countries","currency","city-in-state","family","gram1-adjective-to-adverb","gram2-opposite","gram3-comparative","gram6-nationality-adjective"]
    for key, analogies in all_sections.items():
        if key in needed_list:
            all_sections[key] = analogies

    return all_sections , all_word


topics , all_word = load_file("word-test.v1.txt")

all_word = list(all_word)




def cos_similarity(x, y):
    dot_product = np.dot(x,y)
    norm_x = norm(x)
    norm_y = norm(y)
    cos_sim = dot_product / (norm_x*norm_y)
    return cos_sim



def find_analogy(first_a, first_b, second_a, word_to_vec, all_words):

    try:
        f_a, f_b, s_c = word_to_vec[first_a], word_to_vec[first_b], word_to_vec[second_a]
    except(KeyError):
        return


    max_cosine_sim = 0
    best_word = None


    for w in model.vocab:

        if w in [first_a, first_b, second_a]:
            continue

        u = f_b - f_a + s_c
        v = word_to_vec[w]


        cosine_sim = cos_similarity(u,v)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w


    return best_word



all_word = list(all_word)
total_true = 0
total_ = 0
for key,values in  topics.items():
    for value in values:
        ones = []
        for word in value:
            ones.append(word)
        my_analogy = find_analogy(ones[0], ones[1], ones[2], model, all_word)
        if ones[3] == my_analogy:
            total_true = total_true + 1
        total_ = total_ + 1
        print(ones[3] == my_analogy)


print(total_true)
print(total_)
print(total_true / total_)



##############TASK2 CLASSIFICATION######################

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression
import multiprocessing
import pandas as pd
import string
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

movie_csv = pd.read_csv('tagged_plots_movielens.csv', usecols = ["plot","tag"])

stop_words = ['','when','doesnt','dont','four','three','two','one','i',"especially", 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', "recently",'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very','a','an' ,'s', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

train_ = []
test_ = []
def words_in_text(text):

    words = []
    sentences = text
    table = str.maketrans({key: None for key in string.punctuation})
    try:
        sentences = sentences.translate(table)
        sentences = sentences.split(" ")
        for word in sentences:
            if word in stop_words:
                continue
            else:
                word = stemmer.stem(word)
                words.append(word.lower())
    except(AttributeError):
        return None

    return words

count = 0

print("Reading train data and test data")
for plot,tag in zip( movie_csv.get("plot") , movie_csv.get("tag") ):

    if count <= 2000 :
        plot_t = words_in_text(plot)
        if plot_t == None :
            continue
        train_.append(TaggedDocument(words=plot_t, tags=[tag]))


    else:
        plot_t = words_in_text(plot)
        if plot_t == None:
            continue
        test_.append(TaggedDocument(words=plot_t, tags=[tag]))


    count = count + 1

print("Reading has finished")

print("Doc2Vec model is preapering")
model = Doc2Vec(min_count=30, window=10, vector_size=600, workers=multiprocessing.cpu_count(),
                alpha=0.055, min_alpha=0.00025, dm=0, dm_mean=0, dm_tag_count=0)

model.build_vocab(x for x in train_)
print("Doc2Vec model is training")
model.train(train_, total_examples=len(train_), epochs=10)
print("Model is saving")
model.save('./movieModel.d2v')

print("Getting matricies for train data")
x_train = []
y_train = []

for matricies in train_:
    x_train.append(model.infer_vector(matricies.words, steps=20))
    y_train.append(matricies.tags[0])

y_test = []
x_test = []

print("Getting matricies for test data")
for matricies in test_:
    x_test.append(model.infer_vector(matricies.words, steps=20))
    y_test.append(matricies.tags[0])


print("Logistic Regression is ready to fit")

classifier = LogisticRegression()
classifier.fit(x_train, y_train)
test_prediction = classifier.predict(x_test)
train_prediction = classifier.predict(x_train)

count_true_train = 0
count_train = len(train_prediction)
for tag in range(len(train_prediction)):
    if train_prediction[tag] == y_train[tag]:
        count_true_train = count_true_train + 1


print('Training accuracy',count_true_train/count_train)

count_true_test = 0
count_test = len(test_prediction)
for tag in range(len(test_prediction)):
    if test_prediction[tag] == y_test[tag]:
        count_true_test = count_true_test + 1

print('Testing accuracy ', count_true_test/count_test)

