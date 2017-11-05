import sys
import numpy as np
import pandas as pd
import random

data = []

for i in np.arange(int(random.random()*100)):
	for j in np.arange(int(random.random()*100)):
		para = random.random()*5
		data.append([i, int(random.lognormvariate(para, para))])

data = pd.DataFrame(data, columns=['uid', 'vid'])
data = data[['uid', 'vid']]
with open('data/lognormal.csv', 'a') as f:
   data.to_csv(f, header=True)

data = []

# for i in np.arange(int(random.random()*1000)):
# 	for j in np.arange(int(random.random()*1000)):
# 		para = random.random()*10
# 		data.append([i, int(random.expovariate(1/para))])

# data = pd.DataFrame(data, columns=['uid', 'vid'])
# data = data[['uid', 'vid']]
# with open('data/exponential.csv', 'a') as f:
#    data.to_csv(f, header=True)
