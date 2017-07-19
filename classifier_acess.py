from csv_reader import load_data_access
from sklearn.naive_bayes import MultinomialNB

x, y = load_data_access()
print("------------")
print("x = %s" % x)
print("------------")
print("y = %s" % y)




print("------------")
# training
model = MultinomialNB()
model.fit(x, y)


# teste
test1 = [1, 1, 1]
test2 = [1, 0, 0]
test3 = [0, 0, 0]
test = [test1, test2, test3]
expected = [1, 0, 100]
result = model.predict(x)
diff = result - y
points = [err for err in diff if err == 0]
# resultado do naive bayes
#print "Resultado: %s" % result
# percentual de acertos (conforme esperado)
print "Acertos: %.3f%%" % 	(100.0*len(points)/len(x))	
