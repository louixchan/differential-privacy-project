import sys
import csv
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import errorReporting

# ---
# MAIN PROGRAM
# ---

filename = sys.argv[1]
col1 = sys.argv[2]
col2 = sys.argv[3]
e = pd.to_numeric(sys.argv[4])
iteration = pd.to_numeric(sys.argv[5])
steps = pd.to_numeric(sys.argv[6:])

raw_data = pd.read_csv('data/' + filename + '.csv')
#print(raw_data['source'])
#pre_processed = raw_data.drop('source', 1).groupby(['vid']).agg({'uid':{'frequency':'count'}})
#pre_processed.columns = pre_processed.columns.droplevel(0)
#pre_processed['id'] = np.arange(len(pre_processed))

user_list = raw_data[col1].unique()

#print("There is %d different venue in the file" % len(venue_list))
print("There is %d different users in the file" % len(user_list))

export_header = True;

for step in steps:
    start_time = time.clock()

    novel_mechanism_time = 0

    export = []

    for a in np.arange(iteration):
        for user in user_list:
            pre_processed = raw_data[raw_data[col1]==user]

            print("There is %d different venue for user %d" % (len(pre_processed[col2].unique()), user))

            pre_processed = pd.DataFrame({ 'count' : pre_processed.groupby([col2]).size()}).reset_index()
            pre_processed[col1] = user

            # here we keep only non zero counts
            pre_processed = pre_processed[pre_processed['count'] > 0].sort('count').reset_index()
            bin_count = len(pre_processed)

            novel_start_time = time.clock()
            v_i = np.ceil(bin_count / 2) + (bin_count - 1) * step / 2
            sum_v = bin_count * v_i - bin_count * (bin_count - 1) * step / 2

            e1 = e/2
            pre_processed['masked_count'] = 0.0

            for i in np.arange(bin_count):
                pre_processed.loc[[i], ['masked_count']] = pre_processed.get_value(i, 'count') + np.random.laplace(scale=(sum_v/(v_i * bin_count * e1)))
                v_i = v_i - step
            novel_mechanism_time = novel_mechanism_time + time.clock() - novel_start_time

            pre_processed['masked_count'] = pre_processed['masked_count'].apply(lambda x : x if (x > 0) else 0)
            pre_processed = pre_processed.sort('masked_count').reset_index()
            pre_processed = pre_processed[['level_0', col1, col2, 'count', 'masked_count']]
            pre_processed.columns = ['original_rank', col1, col2, 'count', 'masked_count']

            prefix_sum_array = [0]
            prefix_sum_array.append(pre_processed.get_value(0, 'masked_count'))
            for i in np.arange(bin_count - 1):
                prefix_sum_array.append(pre_processed.get_value(i, 'masked_count') + prefix_sum_array[-1])

            # ---------
            # CLUSTERING
            # ---------

            pre_processed['cluster_id'] = 0
            cluster_means = []

            i = 0
            j = 1

            current_cluster_size = 1
            cluster_means.append(pre_processed.get_value(0, 'masked_count'))
            e2 = e/2

            while j < bin_count:
                include_error = 2/((current_cluster_size + 1) * np.power(e2, 2))
                exclude_error = 2/((current_cluster_size) * np.power(e2, 2))
                
                mean_1 = (prefix_sum_array[j + 1] - prefix_sum_array[j - current_cluster_size])/(current_cluster_size + 1)
                
                for k in np.arange(current_cluster_size + 1) + j - current_cluster_size:
                    include_error = include_error + np.power(pre_processed.get_value(k, 'masked_count') - mean_1, 2)
                
                for k in np.arange(current_cluster_size) + j - current_cluster_size:
                    exclude_error = exclude_error + np.power(pre_processed.get_value(k, 'masked_count') - cluster_means[-1], 2)

                l = 1
                stop = False
                
                while (l < bin_count - j and stop == False):
                    mean_2 = (prefix_sum_array[j + l] - prefix_sum_array[j])/l
                    mean_3 = (prefix_sum_array[j + l + 1] - prefix_sum_array[j])/(l + 1)
                    lhs = np.power(pre_processed.get_value(j, 'masked_count') - mean_3, 2) - np.power(pre_processed.get_value(j, 'masked_count') - mean_2, 2)
                    
                    if (lhs > 2/(l * l * e2 * e2) - 2/(np.power(bin_count - j, 2) * e2 * e2)):
                        exclude_error = exclude_error + np.power(pre_processed.get_value(j, 'masked_count') - mean_2, 2) + 2/(l * l * e2 * e2)
                        stop = True
                    else:
                        l = l + 1

                if (include_error < exclude_error):
                    pre_processed.loc[[j], ['cluster_id']] = i
                    current_cluster_size = current_cluster_size + 1
                    cluster_means[-1] = mean_1
                else:
                    i = i + 1
                    pre_processed.loc[[j], ['cluster_id']] = i
                    current_cluster_size = 1
                    cluster_means.append(pre_processed.get_value(j, 'masked_count'))

                j = j + 1

            pre_processed['budget'] = e
            pre_processed['final_count'] = 0
            pre_processed['squared_error'] = 0
            pre_processed['kld'] = 0
            pre_processed['iteration'] = a

            total_count = pre_processed['count'].sum()

            for i in np.arange(bin_count):
                final_count = cluster_means[pre_processed.get_value(i, 'cluster_id')] + np.random.laplace(scale=(1/e2))
                if (final_count > 0):
                    pre_processed.loc[[i], ['final_count']] = final_count
                
                pre_processed.loc[[i], ['squared_error']] = np.power(pre_processed.get_value(i, 'final_count') - pre_processed.get_value(i, 'count'), 2)

            total_final_count = pre_processed['final_count'].sum()

            for i in np.arange(bin_count):
                if (pre_processed.get_value(i, 'final_count') > 0):
                    pre_processed.loc[[i], ['kld']] = (pre_processed.get_value(i, 'count') / total_count) * np.log((pre_processed.get_value(i, 'count') * total_final_count) / (pre_processed.get_value(i, 'final_count') * total_count))
            # with open('export/NOVEL-' + filename + '-' + str(step) + '-export.csv', 'a') as f:
            #     pre_processed.to_csv(f, header=export_header)
            #     if (export_header):
            #         export_header = False

            
            export.append([filename, e, step, iteration, user, pre_processed['squared_error'].mean(), pre_processed['kld'].sum(), novel_mechanism_time, time.clock() - start_time])
    errorReporting.reportError(pd.DataFrame(export, columns=['dataset', 'budget', 'step', 'iteration', 'id', 'mse', 'kld', 'novel_mechanism_time', 'total_time']))
sys.exit()
