import multiprocessing as mp
import numpy as np
import csv

def z_score(array):
    res_array=np.empty(array[:,0:].shape)
    for i in range(0,len(array[:,0:])):
        vector = array[i, :]
        if i==0:
            res_array=(vector-np.mean(vector))/np.std(vector)
        else:
            vector1=(vector-np.mean(vector))/np.std(vector)
            res_array=np.vstack((res_array,vector1))
    return res_array

def maxs(array):
    res_array = np.empty((array[:, 0:].shape[0], array[:, 0:].shape[1] * 2))
    for i in range(0, len(array[0, :])):
        vector = features_split[:, i]
        mean = np.mean(vector)
        index = np.argmax(vector, axis=0)
        indexer = np.repeat(index, len(features_split[:, i]))
        mae = np.repeat(np.abs(vector[index] - mean), len(features_split[:, i]))
        prelim_res = np.vstack((indexer, mae)).transpose()
        if i == 0:
            res_array = prelim_res
        else:
            res_array = np.concatenate((res_array, prelim_res), axis=1)
    return res_array


with open('train.tsv') as tsvfile:
  reader = csv.reader(tsvfile, dialect='excel-tab')
  data = [data for data in reader]


data_array = np.asarray(data)
id_jobs=data_array[1:,0]
features=data_array[:,1]
features_split=[]


for i in range(1,len(features)):
    features_split.append(features[i].split(","))


features_split=np.asarray(features_split)
res=np.vstack((id_jobs,features_split[:,0]))
features_split=features_split[:,1:].astype(np.int)
means=np.mean(features_split, axis=0)
stds = np.std(features_split,axis=0)
z_scores=z_score(features_split)
res=np.vstack((res,z_scores.transpose()))
maxs=maxs(features_split)
res=np.concatenate((res.transpose(),maxs),axis=1)
feature_names=[]
feature_max_index_names=[]


for i in range(0,len(features_split[0,0:])):
    name="feature_2_stand_"+str(i)
    feature_names.append(name)
    name="max_feature_2_index_"+str(i)
    feature_max_index_names.append(name)
    name="max_feature_2_mean_diff_"+str(i)
    feature_max_index_names.append(name)

names_list=["id_job","features_code"]
names_list=names_list+feature_names+feature_max_index_names
names_list=[names_list]
res=np.concatenate((names_list,res),axis=0)

with open('test_proc.tsv','w+') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    for i in range(0, len(res)):
        writer.writerow(res[i])



