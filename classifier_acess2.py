from csv_reader import load_data_access
from sklearn.naive_bayes import MultinomialNB


x, y = load_data_access()


percentual_to_train = 0.91
total_to_train = int(	round(percentual_to_train*len(x), 2))
total_to_test = len(x) - total_to_train

#
# training
#
print("------training...------")
print("x_training = %s" % x[:total_to_train])
print("y_training = %s" % y[:total_to_train])

model = MultinomialNB()
model.fit(x[:total_to_train], y[:total_to_train])

#
# teste
#
print("------testing...------")
print("x_testing = %s" % x[-1 * total_to_test:])
print("y_testing = %s" % y[-1 * total_to_test:])

result = model.predict(x[-1 * total_to_test:])
diff = result - y[-1 * total_to_test:]

#
# validating
#
print("------validating...------")
points = [err for err in diff if err == 0]
# resultado do naive bayes
#print "Resultado: %s" % result
# percentual de acertos (conforme esperado)

print("Right predictions: %.3f%%" % float(100.0*len(points)/total_to_test))
print("Total trainings: %d" % total_to_train)	
print("Total tests: %d" % total_to_test)	
