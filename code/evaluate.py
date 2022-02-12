#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from ast import literal_eval
import sklearn as sk
import sklearn.metrics
np.random.seed(123)
# %matplotlib inline


# In[22]:


df_source = 'all_exploded'
df_true = pd.read_csv('data/all_exploded/cleaned_expanded_true.csv')
df_fakes = pd.read_csv('data/all_exploded/cleaned_expanded_fakes.csv')

# Do this because data is bad
df_true['speaker_id'] = [f.split('_')[-2] for f in df_true['filepath']]


# In[23]:


len(df_fakes[df_fakes['speaker_id'].isin(df_true['speaker_id'].unique())]['speaker_id'].unique())


# In[24]:


# Create test set
df_test_f = pd.DataFrame([])
df_test_t = pd.DataFrame([])
df_test = pd.DataFrame([])


# In[27]:


len(df_true['speaker_id'].unique()), len(df_fakes['speaker_id'].unique())


# In[28]:


# pick random fake speakers from the for the DF set
all_speakers = df_fakes['speaker_id'].unique()
ixs = random.sample(range(len(all_speakers)), 250)
speakers = df_fakes['speaker_id'].unique()[ixs]

# Add to test set
df_test_f = df_fakes[df_fakes['speaker_id'].isin(speakers)]
df_test_t = df_true[df_true['speaker_id'].isin(speakers)]

# drop that speaker from all the other datasets
df_true = df_true[~df_true['speaker_id'].isin(speakers)]
df_fakes = df_fakes[~df_fakes['speaker_id'].isin(speakers)]


# In[7]:


# pick random oragnic speakers from the for the true set
# all_speakers = df_true['speaker_id'].unique()
# ixs = np.random.randint(0,len(all_speakers), 250)
# speakers = df_true['speaker_id'].unique()[ixs]

# # Add to test set
# df_test_f = df_test_f.append(df_fakes[df_fakes['speaker_id'].isin(speakers)])
# df_test_t = df_test_t.append(df_true[df_true['speaker_id'].isin(speakers)])

# # drop that speaker from all the other datasets
# df_true = df_true[~df_true['speaker_id'].isin(speakers)]
# df_fakes = df_fakes[~df_fakes['speaker_id'].isin(speakers)]


# In[29]:


df_test_t = df_test_t.drop_duplicates()    
df_test_f = df_test_f.drop_duplicates()    
df_test = df_test.append(df_test_f)
df_test = df_test.append(df_test_t)


# In[30]:


len(df_true['speaker_id'].unique()), len(df_fakes['speaker_id'].unique()), len(df_test['speaker_id'].unique())


# In[ ]:


def plot_pdf(data, color, label = None):
    weights = np.ones_like(data)/len(data)
    h = np.histogram(data,bins=50, weights=weights,normed=False,)
    plt.plot(h[1][:-1], h[0], color = color, label = label)
    return h[1], h[0]

def get_best_precision_recall(data_tt, max_val, f):
    for threshold in np.linspace(0, max_val, 25):
        p_val, r_val = get_precision_recall(data_tt, threshold, f)
        if(p_val > 0.9 and r_val > 0.9):
#                 print(p_val, r_val, ' ------- ', threshold)
                return p_val, r_val, threshold
    return -1, -1, -1
        
def get_precision_recall(data_tt, threshold, f):
    y_true = list(data_tt['dataset'] == 'fakes') # set fakes to true
    y_pred = data_tt[f] > threshold # set less than than to true
    
    recall = sk.metrics.recall_score(y_true, y_pred)
    precision = sk.metrics.precision_score(y_true, y_pred)
    return precision, recall


# In[ ]:


# Need val set for precision recall calculation
# The sklearn precision and recall functions take in a two lists
# for y_pred and y_true. So you need to make a val set 
# composed of both true and fakes. (look at get_precision_recall)
df_val = pd.DataFrame()
df_val = df_val.append(df_fakes)
df_val = df_val.append(df_true)


# In[ ]:


len(df_val['speaker_id'].unique())


# In[ ]:





# In[ ]:





# In[ ]:


# Intersection of phonemes
# Phonemes that exist in both true and fakes sets
phonemes = df_val['label'].unique()
features =  df_val.columns[6:21]

ideal_feats = [] # phoneme/feature/threshold/Precision/Recall 
for p in phonemes:
    # Actaully used for precision recall calc
    data_v = df_val[df_val['label'] == p]
    data_f = df_fakes[df_fakes['label'] == p]

    # Used for plots
    data_t = df_true[df_true['label'] == p]
    
    # For each feature 
    for f in features:

        # Get the max value of the feature
        max_val = np.max([np.max(data_t[f]), np.max(data_f[f])])        
        precision, recall, threshold = get_best_precision_recall(data_v, max_val, f)
        
        # only prints if precision and recall are both above 0.95
        if(precision != -1 and recall != -1):
            ideal_feats.append([p, f, threshold, precision, recall])
            
            if(False): # <- Set to True to plot, False to not Plot 
                print('Phoneme = ',p,'         Feature = ',f)

                _, _ = plot_pdf(data_t[f].values, color = 'orange', label ='Organic Audio')
                _, _ = plot_pdf(data_f[f].values, color = 'blue', label ='Deepfake Audio')
                
                a = data_f[f].values

                title = 'PDF for Bigram = '+p
                plt.title(title)
                plt.xlabel('Feature Values')
                plt.ylabel('Probability')
                plt.legend()
                plt.show()
            


# In[ ]:





# In[ ]:





# In[ ]:


df = pd.DataFrame(ideal_feats, columns=['Phoneme', 'Feature', 'Threshold', 'Precision', 'Recall'])
df.to_csv('data/all_exploded/ideal_feats.csv')


# In[ ]:


ideal_feats = np.array([np.array(a) for a in ideal_feats])
dfs = [df_test_f, df_test_t]
final_list = []

for df in dfs:
    
    # Filter speaker so df only contains the important bigram
    total_count_succ = 0
    total_count_fail = 0
    total_files_run = 0
    spkrs = df.speaker_id.unique()

    print('Total Speakers ', len(spkrs))

    for s in spkrs:
        single_spkr = df[df['speaker_id'] == s]
        files = single_spkr.filepath.unique()

        for f in files:
            single_f = single_spkr[single_spkr['filepath'] == f]
            single_f = single_f[single_f.label.isin(ideal_feats[:,0])]

            succ = 0
            fail = 0
            bigram_count = 0
            for i in ideal_feats:
                ph = i[0]
                f = i[1]
                th = i[2]

                single = single_f[single_f['label'] == ph]
                
                if(len(single) != 0):
                    bigram_count +=1
                    succ = succ + np.count_nonzero(single[f]>=float(th))
                    fail = fail + np.count_nonzero(single[f]<float(th))
            
            label = 'fakes' if(succ > fail) else 'True'

            if(succ  != 0 and fail != 0):
                final_list.append([single_f['dataset'].values[0], label, bigram_count, succ, fail])
    
    final_list_tmp = np.asarray(final_list)
    y_pred = final_list_tmp[:,1]
    y_true = final_list_tmp[:,0]
    
    print('Success Rate for detecting', y_true[0],':', np.count_nonzero(y_pred == y_true)/len(y_true))


# In[ ]:


final_list = np.array([np.array(a) for a in final_list])
final_list_pd = pd.DataFrame(final_list, columns =['original_label', 'our_label', 'ideal_feats', 'pos', 'neg'])
final_list_pd.to_csv('data/all_exploded/classification_results.csv')


# In[ ]:



print('Speakers Eval')
print(df_val['speaker_id'].unique())

print('Speakers Test')
print(df_test_f['speaker_id'].unique())
print(df_test_t['speaker_id'].unique())


# In[31]:


get_ipython().system('jupyter nbconvert all_expanded.ipynb --to script ')


# In[ ]:





