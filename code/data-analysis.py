import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---
# MAIN PROGRAM
# ---

filename = sys.argv[1]

raw_data = pd.read_csv(filename)
#print(raw_data['source'])
#pre_processed = raw_data.drop('source', 1).groupby(['vid']).agg({'uid':{'frequency':'count'}})
#pre_processed.columns = pre_processed.columns.droplevel(0)
#pre_processed['id'] = np.arange(len(pre_processed))

user_list = raw_data.uid.unique()

#print("There is %d different venue in the file" % len(venue_list))
print("There is %d different users in the file" % len(user_list))

export_header = True

for user in user_list:
    pre_processed = raw_data[raw_data['uid']==user]

    print("There is %d different venue for user %d" % (len(pre_processed.vid.unique()), user))

    pre_processed = pd.DataFrame({ 'count' : pre_processed.drop(['source'], 1).groupby(['vid']).size()}).reset_index()
    pre_processed['uid'] = user
    
    # here we keep only non zero counts
    pre_processed = pre_processed[pre_processed['count'] > 0].sort('count').reset_index()
    bin_count = len(pre_processed)

    #plt.figure(figsize=(20,10))
    #plt.bar(np.arange(len(pre_processed)) - 0.25, pre_processed, 0.5, alpha=0.5, color='red', label='check-in frequency')
    #plt.gca().yaxis.grid()
    #plt.xlim(xmin = -1, xmax = len(pre_processed))
    #plt.ylabel("Check-in Frequency")
    #plt.xlabel("Venue ID")
    #plt.title("Original Histogram")
    #plt.savefig('original.png')
    #plt.legend()
    #plt.show()

    e1 = 0.05
    e2 = 0.05
    
    pre_processed['masked_count'] = 0.0

    for i in np.arange(bin_count):
        pre_processed.loc[[i], ['masked_count']] = pre_processed.get_value(i, 'count') + np.random.laplace(scale=(1/e1))
    
    bin_count = len(pre_processed)
    threshold = (0 * np.log(bin_count)/e1)
    
    pre_processed['masked_count'] = pre_processed['masked_count'].apply(lambda x : x if (x > threshold) else 0)
    pre_processed = pre_processed.sort('masked_count').reset_index()
    pre_processed = pre_processed[['uid', 'vid', 'count', 'masked_count']]
    pre_processed.columns = ['uid', 'vid', 'count', 'masked_count']

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
    e2 = 0.05
    
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
    
        print ("j: %d" % j)
        
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

    pre_processed['final_count'] = 0
    pre_processed['squared_error'] = 0
    
    for i in np.arange(bin_count):
        final_count = cluster_means[pre_processed.get_value(i, 'cluster_id')] + np.random.laplace(scale=(1/e2))
        if (final_count > 0):
            pre_processed.loc[[i], ['final_count']] = final_count
        
        pre_processed.loc[[i], ['squared_error']] = np.power(pre_processed.get_value(i, 'final_count') - pre_processed.get_value(i, 'count'), 2)

    with open('export/AHP-export.csv', 'a') as f:
        pre_processed.to_csv(f, header=export_header)
        if (export_header):
            export_header = False

sys.exit()
