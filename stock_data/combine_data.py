import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
import csv
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import datetime


file_ = open('stock_price_direction_all.csv', 'w')
csvCursor = csv.writer(file_)

count = 0
total = 0
for i in range(2000,2019):
	file = open('stock_price_direction_'+str(i)+'.csv', 'r')
	csvCursor_r = csv.reader(file)

	j = 0
	for row in csvCursor_r:
		if j == 0:
			pass
		else:
			csvCursor.writerow(row)
			count += 1
			total += float(row[5])
		j += 1


mean = total/count

file_ = open('stock_price_direction_all_new.csv', 'w')
csvCursor = csv.writer(file_)


file = open('stock_price_direction_all.csv', 'r')
csvCursor_r = csv.reader(file)
for row in csvCursor_r:
	row[5] = (float(row[5]) / mean)-1
	csvCursor.writerow(row)



file_ = open('stock_price_direction_all_new2.csv', 'w')
csvCursor = csv.writer(file_)

file = open('stock_price_direction_all_new.csv', 'r')
csvCursor_r = csv.reader(file)
for row in csvCursor_r:
	row.append(row[8])
	row[8] = 1 if row[8]=='u10' else -1
	csvCursor.writerow(row)

