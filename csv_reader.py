# -*- coding: UTF-8 -*- 
import sys
import traceback
import csv

def load_data_access():
	file = None
	x = []
	y = []
	try:
		file = open("data_page_access.csv", "r")
		reader = csv.reader(file)
		reader.next()
		for home,como_funciona,contato,comprou in reader:
			x.append([int(home),int(como_funciona),int(contato)])
			y.append(int(comprou))
	except IOError as err:
		print "IOError: %s" % err
	except Exception as err:
		print "------------------------------------------------------------------------------------"
		print "Unexpected error:", sys.exc_info()[0]
		print 	"------------------------------------------------------------------------------------"
		traceback.print_exc()
		print "------------------------------------------------------------------------------------"
	finally:
		if file is not None:
			file.close()

	return x, y

def load_data_search():
	file = None

	x = []
	y = []

	try:
		file = open("data_page_search.csv", "r")
		reader = csv.reader(file)
		reader.next()

		for home,busca,logado,comprou in reader:
			print "l"
			x.append([int(home), busca, int(logado)])
			y.append(int(comprou))
	except Exception as err:
		print "------------------------------------------------------------------------------------"
		print "Unexpected error:", sys.exc_info()[0]
		print 	"------------------------------------------------------------------------------------"
		traceback.print_exc()
		print "------------------------------------------------------------------------------------"
	finally:
		if file is not None:
			file.close()

	return x, y

