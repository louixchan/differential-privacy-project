import sys
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

def reportError(raw):
    # ---
    # MAIN PROGRAM
    # ---

    export_error = pd.DataFrame({'id':[]})

    summary = raw.groupby(['dataset', 'budget', 'step', 'id']).agg({'mse': 'mean', 'kld': 'sum', 'novel_mechanism_time': 'mean', 'total_time': 'mean'}).reset_index()
    summary['iteration'] = np.max(raw['iteration'])
    summary['time'] = datetime.datetime.now()
    summary = summary[['time', 'dataset', 'budget', 'step', 'iteration', 'mse', 'kld', 'novel_mechanism_time', 'total_time']]
    summary['novel_percentage'] = summary['novel_mechanism_time'] / summary['total_time']

    with open('export/error-details.csv', 'a') as f:
        summary.to_csv(f, header=False)

    with open('export/error-summary.csv', 'a') as f:
        summary = summary.groupby(['time', 'dataset', 'budget', 'step']).agg({"iteration": "max", "mse": {"mse-mean": "mean", "mse-max": "max", "mse-min":  "min"}, "kld": {"kld-mean": np.mean, "kld-max": "max", "kld-min":  "min"}, "novel_mechanism_time": "sum", "total_time": "sum"}).reset_index()
        summary['novel_percentage'] = summary['novel_mechanism_time'] / summary['total_time']
        summary.to_csv(f, header=False)

    # with open('export/error-summary.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(export_summary)

    # print(export_error)
    #with open('export/error-' + path + '-export.csv', 'a') as f:
    #    export_error.to_csv(f, header=True)
