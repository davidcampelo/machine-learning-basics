# -*- coding: UTF-8 -*- 
#from csv_reader import load_data_search
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from collections import Counter

def load_data():
	dataframe = pd.read_csv("data_page_search2.csv")

	x_dataframe = dataframe[['home','busca','logado']]
	x_dummies = pd.get_dummies(x_dataframe).astype(int)
	x = x_dummies.values

	y_dataframe = dataframe['comprou']
	y_dummies = y_dataframe
	y = y_dummies.values

	percentual_to_train = 0.8
	percentual_to_test = 0.1

	total_to_train = int(	round(percentual_to_train*len(x), 2))
	total_to_test = int(	round(percentual_to_test*len(x), 2))
	total_to_validate = len(x) - total_to_train - total_to_test

	x_to_train    = x[0:total_to_train]
	y_to_train    = y[0:total_to_train]
	x_to_test     = x[total_to_train:total_to_train+total_to_test]
	y_to_test     = y[total_to_train:total_to_train+total_to_test]
	x_to_validate = x[-1 * total_to_validate:]
	y_to_validate = y[-1 * total_to_validate:]

	print("------Data information------")
	print("Total trainings   = %d" % len(x_to_train))	
	print("Total tests       = %d" % len(x_to_test))	
	print("Total validate    = %d" % len(x_to_validate))	
	#
	# dummy prediction = always 1 or 0
	# the one which happens most will be the minimum threshold (using the test data!)
	# create a counter for the test vars and use the which happens the most
	counter = max(Counter(y_to_test).itervalues())
	print("Minimum threshold = %.2f%%" % (100.0 * counter/len(y_to_test)))

	return x_to_train, y_to_train, x_to_test, y_to_test, x_to_validate, y_to_validate

def fit(model, x_to_train, y_to_train):
	model.fit(x_to_train, y_to_train)

def predict(model, x_to_test, y_to_test):

	result = model.predict(x_to_test)
	diff = (result == y_to_test)

	points = sum(diff)
	result = 100.0*points/len(y_to_test)
	print("Total right predictions of [%s] = %.2f%%" % (type(model).__name__, result))

	return result


		
x_to_train, y_to_train, x_to_test, y_to_test, x_to_validate, y_to_validate = load_data()

print("------Predictions with some algorithms------")
modelMNB = MultinomialNB()
fit(modelMNB, x_to_train, y_to_train)
resultMNB = predict(modelMNB, x_to_test, y_to_test)

modelABC = AdaBoostClassifier()
fit(modelABC, x_to_train, y_to_train)
resultABC = predict(modelABC, x_to_test, y_to_test)


print("------Validation with the best algorithm------")
results = {}
results[resultMNB] = modelMNB
results[resultABC] = modelABC

# Best model has the higher result 
modelBest = results[max(results)]

#
# validate agaist the last part of the data
predict(modelBest, x_to_validate, y_to_validate)