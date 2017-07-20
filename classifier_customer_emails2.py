#! -*- coding:utf-8 -*-

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


import pandas as pd
import nltk

from classifier import fit_model
from classifier import validate_model
from classifier import prepare_data
from classifier import minimum_threshold

K = 10
PERCENTUAL_TO_sample = 0.8
MINIMUM_WORD_LENGTH = 1

#
# download NLTK packages
#
# nltk.download('stopwords')
# nltk.download('rslp')
# nltk.download('punkt')
PORTUGUESE_STOP_WORDS = nltk.corpus.stopwords.words("portuguese")
STEMMER = nltk.stem.RSLPStemmer()

# read file
datafile = pd.read_csv("data_emails.csv", encoding="utf-8")
# take email column only
raw_emails = datafile['email'].str.lower()


# tokenize all email lines - separate words considering punctuation
tokenized_emails = [nltk.tokenize.word_tokenize(email_line) for email_line in raw_emails]

# put all words in a set (unique values)
# put only word stems
# ignore stop words (common words with no important meaning - e.g. prepositions, pronouns)

words = set()
for email_array in tokenized_emails:
	valid_words = [STEMMER.stem(word) for word in email_array if word not in PORTUGUESE_STOP_WORDS and len(word) > MINIMUM_WORD_LENGTH]
	words.update(valid_words)


# create a dictionary of words => index
# create a dictionary of index => words (reverse) - for debugging (!)
number_of_words = len(words)
tuples = zip(words, xrange(number_of_words))
dictionary = {word: index for word, index in tuples}
reverse_dictionary = {index: word for word, index in tuples}


# for each line, create a vector of words and count every word occurrence
def vectorize_line(email_array, dictionary, stemmer):
	vectorized = [0] * len(dictionary)
	for word in email_array:
		if len(word) > MINIMUM_WORD_LENGTH:
			word_stem = stemmer.stem(word)
			if word_stem in dictionary:
				index = dictionary[word_stem]
				vectorized[index] += 1

	return vectorized


words_occurrences = [vectorize_line(email_array, dictionary, STEMMER) for email_array in tokenized_emails]
print("\n------Dictionary information------")
print("Total words   = %d" % len(dictionary))

# dump words and lines showing indexes
#
# for i,v in enumerate(words_occurrences[0]):
# 	if v is not 0:
# 		print "{0} = {1} == {2}".format(i, v, reverse_dictionary[i])

# load data for classifier
print("\n------Data loading information------")

x = words_occurrences
y = datafile['classificacao']

x_to_sample, y_to_sample, x_to_validate, y_to_validate = prepare_data(x, y, PERCENTUAL_TO_sample)
print("Total sampleings   = %d" % len(x_to_sample))
print("Total validate    = %d" % len(x_to_validate))

minimum = minimum_threshold(y_to_validate)
print("Minimum threshold = %.4f%%" % minimum)

print("\n------Predictions with k-fold------")
modelOVR = OneVsRestClassifier(LinearSVC(random_state=0))
resultOVR = fit_model(modelOVR, x_to_sample, y_to_sample, K)
print("Total right predictions of [%s] = %.4f%%" % (type(modelOVR).__name__, resultOVR))

modelOVO = OneVsOneClassifier(LinearSVC(random_state=0))
resultOVO = fit_model(modelOVO, x_to_sample, y_to_sample, K)
print("Total right predictions of [%s] = %.4f%%" % (type(modelOVO).__name__, resultOVO))

modelMNB = MultinomialNB()
resultMNB = fit_model(modelMNB, x_to_sample, y_to_sample, K)
print("Total right predictions of [%s] = %.4f%%" % (type(modelMNB).__name__, resultMNB))

modelABC = AdaBoostClassifier()
resultABC = fit_model(modelABC, x_to_sample, y_to_sample, K)
print("Total right predictions of [%s] = %.4f%%" % (type(modelABC).__name__, resultABC))

modelDTC = DecisionTreeClassifier()
resultDTC = fit_model(modelDTC, x_to_sample, y_to_sample, K)
print("Total right predictions of [%s] = %.4f%%" % (type(modelDTC).__name__, resultDTC))


print("\n------Validation with the best algorithm------")
results = {}
results[resultOVR] = modelOVR
results[resultOVO] = modelOVO
results[resultMNB] = modelMNB
results[resultABC] = modelABC
results[resultDTC] = modelDTC
# Best model has the higher result value
modelBest = results[max(results)]

#
# validate against the last part of the data
#
result = validate_model(modelBest, x_to_sample, y_to_sample, x_to_validate, y_to_validate)
print("Total right validations of [%s] = %.4f%%" % (type(modelBest).__name__, result))

