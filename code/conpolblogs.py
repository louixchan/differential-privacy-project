import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

# ---
# MAIN PROGRAM
# ---

path = "data/conpolblogs.ncol"
    
raw = pd.read_csv(path)

with open('data/conpolblogs-out.csv', 'a') as f:
   raw.groupby(['out']).count().to_csv(f, header=True)
