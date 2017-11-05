import pandas as pd
import numpy as np
import sys

export=pd.read_csv('data/lognormal.csv').groupby(['uid']).agg({
		'vid' : 'count'
	})

print(export)
print(np.mean(export['vid']))
print(np.var(export['vid']))
print(np.min(export['vid']))
print(np.max(export['vid']))