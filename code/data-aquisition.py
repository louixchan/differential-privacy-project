import sys
import numpy as np
import csv
import pandas as pd

#### MAIN PROGRAM ####

filename = "data/" + sys.argv[1]
csvfilename = "data/" + sys.argv[1][:-4] + ".csv"

try:
    with open(filename, "r") as f:
        data = pd.read_csv(filename, sep='\t', names = ["uid", "vid"])

        data = data[["uid", "vid"]]
        data["source"] = sys.argv[1]
        data.to_csv(csvfilename, sep=',')
except IOError as iox:
    print("Error opening " + filename + " : " + str(iox))
    sys.exit()

numrec = len(data)
print ("%s contains %d rows of data" %(filename, numrec))
print (data)


#for rec in data:
#    temp = rec.split(' ')
#    print(temp[0]),
#    print(temp[1])
