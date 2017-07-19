# -*- coding: UTF-8 -*- 
#from csv_reader import load_data_search
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

dataframe = pd.read_csv("data_page_search.csv")

x_dataframe = dataframe[['home','busca','logado']]
x_dummies = pd.get_dummies(x_dataframe)
x = x_dummies.values

y_dataframe = dataframe['comprou']
y_dummies = y_dataframe
y = y_dummies.values


percentual_to_train = 0.9
total_to_train = int(	round(percentual_to_train*len(x), 2))
total_to_test = len(x) - total_to_train

#
# training
#
print("------training------")
print("x_training = %s = %d" % (x[:total_to_train], len(x[:total_to_train])))
print("y_training = %s  = %d" % (y[:total_to_train], len(y[:total_to_train])))

	
model = MultinomialNB()
model.fit(x[:total_to_train], y[:total_to_train])

#
# teste
#
print("------testing------")
print("x_testing = %s = %d" % (x[-1 * total_to_test:], len(x[-1 * total_to_test:])))
print("y_testing = %s = %d" % (y[-1 * total_to_test:], len(y[-1 * total_to_test:])))

result = model.predict(x[-1 * total_to_test:])
diff = result - y[-1 * total_to_test:]

#
# validating
#
print("------validating------")
points = [err for err in diff if err == 0]

#
# dummy prediction = always 1 or 0
# the one which happens most will be the minimum threshold
# 
test_one = len(y[y==1]) # sum all 1
test_zero = len(y[y==0]) # sum all 0

print("------results------")
print("Minimum threshold: %.2f%%" % (100.0 * max(test_zero, test_one)/len(y)))
print("Total right predictions: %.2f%%" % (100.0*len(points)/total_to_test))
print("Total trainings: %d" % total_to_train)	
print("Total tests: %d" % total_to_test)	
