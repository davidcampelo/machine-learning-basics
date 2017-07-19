# -*- coding: UTF-8 -*- 
#from csv_reader import load_data_search
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score

import pandas as pd

from collections import Counter

import numpy as np

K = 10
PERCENTUAL_TO_TRAIN = 0.9


def load_data():
	dataframe = pd.read_csv("data_customer_status.csv")

	x_dataframe = dataframe[['recencia','frequencia','semanas_de_inscricao']]
	x_dummies = pd.get_dummies(x_dataframe).astype(int)
	x = x_dummies.values

	y_dataframe = dataframe['status']
	y_dummies = y_dataframe
	y = y_dummies.values

	total_to_train = int(	round(PERCENTUAL_TO_TRAIN*len(x), 2))
	total_to_validate = len(x) - total_to_train

	x_to_train    = x[0:total_to_train]
	y_to_train    = y[0:total_to_train]
	x_to_validate = x[total_to_train:]
	y_to_validate = y[total_to_train:]

	print("Total trainings   = %d" % len(x_to_train))	
	print("Total validate    = %d" % len(x_to_validate))	
	#
	# dummy prediction = always 1 or 0
	# the one which happens most will be the minimum threshold (using the test data!)
	# create a counter for the test vars and use the which happens the most
	counter = max(Counter(y_to_validate).itervalues())
	print("Minimum threshold = %.4f%%" % (100.0 * counter/len(y_to_validate)))

	return x_to_train, y_to_train, x_to_validate, y_to_validate

def fit(model, x_to_train, y_to_train):
	results = cross_val_score(model, x_to_train, y_to_train, cv = K)
	mean_results = np.mean(results)*100.0
	print("Total right predictions of [%s] = %.4f%%" % (type(model).__name__, mean_results))
	return mean_results

def validate(model, x_to_train, y_to_train, x_to_validate, y_to_validate):
	model.fit(x_to_train, y_to_train)
	result = model.predict(x_to_validate)
	diff = (result == y_to_validate)

	points = sum(diff)
	result = 100.0*points/len(y_to_validate)
	print("Total right validations of [%s] = %.2f%%" % (type(model).__name__, result))

	return result


print("\n------Data loading information------")
x_to_train, y_to_train, x_to_validate, y_to_validate = load_data()

print("\n------Predictions with k-fold------")
modelOVR = OneVsRestClassifier(LinearSVC(random_state = 0))
resultOVR = fit(modelOVR, x_to_train, y_to_train)

modelOVO = OneVsOneClassifier(LinearSVC(random_state = 0))
resultOVO = fit(modelOVO, x_to_train, y_to_train)

modelMNB = MultinomialNB()
resultMNB = fit(modelMNB, x_to_train, y_to_train)

modelABC = AdaBoostClassifier()
resultABC = fit(modelABC, x_to_train, y_to_train)


print("\n------Validation with the best algorithm------")
results = {}
results[resultOVR] = modelOVR
results[resultOVO] = modelOVO
results[resultMNB] = modelMNB
results[resultABC] = modelABC

# Best model has the higher result 
modelBest = results[max(results)]

#
# validate agaist the last part of the data

validate(modelBest, x_to_train, y_to_train, x_to_validate, y_to_validate)