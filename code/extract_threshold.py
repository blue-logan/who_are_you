"""
Author: Logan Blue
Date: May 18, 2021

For the USENIX 2021 Major revision prcess. We need to change up a few things
interally, and just using Hadi's code is probably going to cause more problems
that just rewriting it all.

So this code will find thresholds for all the bigrams a human speaker is likely
to create (basically a range of values that is realistic) and then saves them.
It will also check (if we have synthetic examples of that bigram) if the
threshold is an ideal classifier. i.e., can we achieve a 90% recall and
precision rate using just that feature for our evaluation set. If so, we can
mark it as an ideal feature for future examples from this model.  """

#pylint: disable='trailing-whitespace', 'invalid-name', 'too-many-locals'
#pylint: disable='pointless-string-statement', 'too-many-statements'
#pylint: disable='singleton-comparison'
import random
import sys, pickle
from multiprocessing import Pool

import pdb
import pymongo
import numpy as np
import pandas as pd
from tqdm import tqdm

import sklearn as sk 
import sklearn.metrics

#turn off warnings
import warnings
warnings.simplefilter('ignore')


#turn off pandas warning
pd.options.mode.chained_assignment = None  # default='warn'

#set fixed random seed for consistency
np.random.seed(123)

#load data
def load_data(table_name, collection_name='exploration'):
    """load the data from a specified table for analysis"""
    myclient = pymongo.MongoClient('mongodb://localhost:27017')
    db = myclient[collection_name]
    table = db[table_name]

    return pd.DataFrame(list(table.find()))

def process_df(df):
    """This function will clean up my dataframe and explore out the different
    cross_sect_est into their own rows"""
    df = df.drop(columns=['_id'])

    #explode data
    df_exp = df.explode('cross_sect_est')

    #add indices in the cross_sect_est
    indices = []
    for _, row in df.iterrows():
        areas = row['cross_sect_est']
        indices += list(range(0, len(areas)))

    df_exp['area_index'] = indices

    return df_exp

def get_organic_ranges(df):
    """ === Implementation for operation 1 ===
    This function will return the ranges win which each organic bigram will 
    be present. 
    Input: df           - Dataframe with the cross-sectional area estimates for
                            organic audio samples. 
    Output: df_ranges   - Dataframe containing all unique bigrams and their 
                            associated ranges
    """
    #get ranges for each organic
    df_ranges = []
    for key, grp in df.groupby(by=['label', 'area_index']):
        #get max and min value for each group with certain key
        df_ranges.append([key[0], key[1], grp.cross_sect_est.min(),
                grp.cross_sect_est.max()])
    
    df_ranges = pd.DataFrame(df_ranges, columns=['label', 'area_index', 'min',\
            'max'])

    return df_ranges

def pooled_get_optimal_threshold(input_values):
    """This is a function that encapsilate much of the same code originally in 
    get_optimal_threshold that has been move so that I can increase processing
    time using a pool. 
    """
    #extract data segments
    df, df_ranges, index, row = input_values

    #sweep through range of the possible values, calculating the recall and 
    #precision values
    for threshold in np.linspace(row['min'], row['max'], 25):
        #output values
        new_row = row.values
        new_row = np.append(new_row, [-1, -1, -1])
        #Hadi's code I'm using
        #set fakes to true
        y_true = list(df[(df.label == row.label) &\
                (df.area_index == row.area_index)]['dataset'] == 'fakes') 
        y_pred = list(df[(df.label == row.label) &\
                (df.area_index == row.area_index)].cross_sect_est >\
                threshold)

        recall = sklearn.metrics.recall_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)

        #augment df with new values: threshold, ideal indicator, precision, recall
        if precision >= 0.9 and recall >= 0.9:
            #save results
            new_row[-4] = True
            new_row[-3] = threshold
            new_row[-2] = recall
            new_row[-1] = precision
            
            #break out since we are done
            break
        
    #we have to return out segment of df_ranges since it isn't shared 
    #between workers
    return new_row

def get_optimal_threshold(df, df_ranges):
    """This function will find the optimal threshold for every bigram within our
    dataset, labeling it as an ideal theshold only is it can achieve a precision
    and recall of at least 0.9"""
    #for every phoneme and area_index
    df_ranges['ideal_feature'] = False

    #create groups to pool over
    data_values = [(df, df_ranges, index, row) for index, row in \
            df_ranges.iterrows()]

    updates = []
    with Pool(4) as p:
        for new_row in tqdm(\
                p.imap_unordered(pooled_get_optimal_threshold, data_values), 
                total=len(data_values), desc='Getting ideal set'):
            updates.append(new_row)
    
    return pd.DataFrame(updates, columns=['label', 'area_index', 'min', 'max',
        'ideal_feature', 'threshold', 'recall', 'precision'])

def calc_non_opt_sentence_threshold(df_ranges, df_data):
    """This function will find maximum error rate floor necessary for our 
    non-optimized detector to function accurately while hopefully minimizing the 
    number of false positives. For example, if we require 5% of all bigrams in a 
    sentense to be positive (and maintain a 100% TPR or recall) then we have 5%
    wiggle room to not get all organic speakers in our extraction set. If we
    move to 6% and our recall goes to less than 100%, then we are trading 
    recall for a better false positive rate."""
    threshold_max = -1
    threshold_min = -1
    threshold_either = -1

    #merge two datasets on label and area_index columns
    df_analysis = pd.merge(df_ranges, df_data, how='right', on=['label', 
            'area_index'])

    #find and mark row that fall outside the organic ranges calculated
    df_analysis['breaks_max'] = df_analysis.apply(lambda row: row['max'] <\
            row.cross_sect_est, axis=1)
    df_analysis['breaks_min'] = df_analysis.apply(lambda row: row['min'] >\
            row.cross_sect_est, axis=1)
    df_analysis['breaks_either'] = df_analysis.apply(lambda row: (row['min'] >\
            row.cross_sect_est) or (row['max'] < row.cross_sect_est), axis=1)


    #per sentence, calculate the percentage of the time we are outside of 
    #organic ranges
    df_results = df_analysis.groupby(['filepath', 'dataset']).agg('mean')
    df_results.reset_index(inplace=True)

    #sweep values through df_results to find a threshold that divides 
    #fakes and organic will enough for the MAX values
    space = np.linspace(0.004, 0.2, 100)
    y_true = list(df_results.dataset == 'fakes')
    for threshold in space:
        y_pred = list(df_results.breaks_max > threshold)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)

        if recall >= 0.9 and precision >= 0.9:
            #good results
            print("Sentence Threshold: ", threshold)
            threshold_max = threshold
            break

    #sweep values through df_results to find a threshold that divides 
    #fakes and organic will enough for the MIN values
    space = np.linspace(0.039, 0.2, 100)
    y_true = list(df_results.dataset == 'fakes')
    for threshold in space:
        y_pred = list(df_results.breaks_min > threshold)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)

        if recall >= 0.9 and precision >= 0.9:
            #good results
            print("Sentence Threshold: ", threshold)
            threshold_min = threshold
            break
    
    #sweep values through df_results to find a threshold that divides 
    #fakes and organic will enough for the MIN values
    space = np.linspace(0.2, 0, 100)
    y_true = list(df_results.dataset == 'fakes')
    for threshold in space:
        y_pred = list(df_results.breaks_either > threshold)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)

        if recall >= 0.9 and precision >= 0.9:
            #good results
            print("Sentence Threshold: ", threshold)
            threshold_min = threshold
            break

    #we didn't find a suitable value
    return threshold_max, threshold_min, threshold_either

def non_opt_test_sentences(df_ranges, df_data, threshold_max, threshold_min, 
        threshold_either):
    """This function will be used to evaluate the effectiveness of our threshold
    on the validation data set. df_ranges are the acceptable organic ranges, 
    df_data is the data we will be processing, and threshold is the percentage
    of any given sentence that can be outside our threshold before we label the
    whole sentence as a deepfake."""
    #merge two datasets on label and area_index columns
    df_analysis = pd.merge(df_ranges, df_data, how='right', on=['label', 
            'area_index'])

    #find and mark row that fall outside the organic ranges calculated
    df_analysis['breaks_max'] = df_analysis.apply(lambda row: row['max'] <\
            row.cross_sect_est, axis=1)
    df_analysis['breaks_min'] = df_analysis.apply(lambda row: row['min'] >\
            row.cross_sect_est, axis=1)
    df_analysis['breaks_either'] = df_analysis.apply(lambda row: (row['max'] <\
            row.cross_sect_est) or (row['min'] > row.cross_sect_est), axis=1)

    #per sentence, calculate the percentage of the time we are outside of 
    #organic ranges
    df_results = df_analysis.groupby(['filepath', 'dataset']).agg('mean')
    df_results.reset_index(inplace=True)

    #mark those that above our detection threshold
    df_results['max_pred'] = df_results['breaks_max'] > threshold_max
    df_results['min_pred'] = df_results['breaks_min'] > threshold_min
    df_results['either_pred'] = df_results['breaks_either'] > threshold_either

    return df_results

        
def opt_test_sentence(df_thresholds, df_data):
    """This function will only examine ideal features when determining if a 
    sentence is organic or not. It will then using the resulting decisions as 
    votes in determining the overall sentences label."""
    #filter out non_ideal features
    print("Starting op testing...")
    df_ideal = df_thresholds[df_thresholds.ideal_feature][['label',\
            'area_index', 'threshold']]
    df_examine = pd.merge(df_ideal, df_data, how='left', on=['label',\
            'area_index'])

    #test all ideal features
    print("Doing apply")
    df_examine['classification'] = df_examine.apply(lambda row: \
            row.cross_sect_est > row.threshold, axis=1)
    
    df_results = []
    groups = df_examine.groupby('filepath')
    #every sentences together (filepath)
    for key, grp in tqdm(groups, desc='voting'):
        #vote
        voting_total_percentage = grp.classification.mean()
        sentence_truth = grp.head(1).dataset.item()
        
        if voting_total_percentage >= 0.5:
            #label sentence as a deepfake
            df_results.append([key, True, sentence_truth])
        else:
            #label sentence as an organic sample
            df_results.append([key, False, sentence_truth])
        
    #save results to a new df
    return pd.DataFrame(df_results, columns=['filepath', 'prediction',\
            'ground_truth'])

def main():
    """This code will be generating our output data values for the guesswho
    project. It was a one point Hadi's code, however, I have gone ahead and
    rewritten it for clarity and small methodological changes. 

    Operations list:
        1) Get all ranges for organic speakers in TIMIT dataset
        2) Check for how well ranges differentiates organic and synthetic audio
            samples. (RESULT)
        3) Extract ideal feature set
            3a) Recalc Hadi's orginal numbers from the paper
        4) Check ASV Spoof and Lyrebird against the ranges found in the TIMIT 
            dataset

    """

    #""
    print("Loading datasets...")
    #start by load audio sets
    print("TIMIT...")
    #TIMIT
    df_timit_true = load_data('timit_true_extended', collection_name='windows')
    df_timit_true = process_df(df_timit_true)
    df_timit_fakes = load_data('real_time_extended', collection_name='windows')
    df_timit_fakes = process_df(df_timit_fakes)


    #add dataset label
    df_timit_true['dataset'] = 'true'
    df_timit_fakes['dataset'] = 'fakes'

    #NOTE: Add additional idiosyncrasies

    #into a single df for analysis
    df_timit = pd.concat([df_timit_true, df_timit_fakes], ignore_index=True)
    df_timit.reset_index(drop=True, inplace=True)

    print()
    print('Loading Complete')

    #create exploration dataset and evaluation datasets
    eval_speaker = random.sample(list(df_timit.speaker_id.unique()), 250)
    df_test = df_timit[~df_timit.speaker_id.isin(eval_speaker)] #small
    df_eval = df_timit[df_timit.speaker_id.isin(eval_speaker)]  #big

    #==========_Operation #1_==========
    #filter out organic audio samples
    #call processor function
    print("Getting organic ranges...")
    df_org_ranges = get_organic_ranges(df_test[df_test.dataset == 'true'])

    #==========_Operation #2_==========
    #find number of bigrams outside of organic range for my deepfakes (zero 
    #organic should, but double check it here). Used to find a threshold value
    #using the exploration set
    print("calc non opt sentence threshold...")
    sentence_threshold_max, sentence_threshold_min, sentence_threshold_either =\
            calc_non_opt_sentence_threshold(df_org_ranges, df_test)

    if sentence_threshold_max < 0 or sentence_threshold_min < 0:
        print('Problems...')
        pdb.set_trace()

    #with threshold, use validation threshold to get preformance of technique
    #TODO: REMOVE TEST
    #df_eval_test = df_eval[df_eval.filepath.isin(
    #        random.sample(set(df_eval.filepath.unique()), 20))]
    #END
    print("Starting non_opt test...")
    #non_opt_test_sentences(df_org_ranges, df_eval, sentence_threshold)
    df_results = non_opt_test_sentences(df_org_ranges, df_eval,\
            sentence_threshold_max, sentence_threshold_min, 
            sentence_threshold_either)

    #TODO: REMOVE
    #save to speed up dev
    #pickle.dump(df_results, open('df_results.pkl', 'wb'))
    #pickle.dump(df_org_ranges, open('org_range.pkl', 'wb'))
    #pickle.dump(df_eval, open('df_eval.pkl', 'wb'))
    #pickle.dump(df_test, open('df_test.pkl', 'wb'))
    #pickle.dump((sentence_threshold_max, sentence_threshold_min),\
    #        open('sent_thres.pkl', 'wb'))
    #""

    #resume code
    #with open('org_range.pkl', 'rb') as f:
    #    df_org_ranges = pickle.load(f)

    #with open('df_eval.pkl', 'rb') as f:
    #    df_eval = pickle.load(f)

    #with open('df_test.pkl', 'rb') as f:
    #    df_test = pickle.load(f)

    #with open('sent_thres.pkl', 'rb') as f:
    #    sentence_threshold_max, sentence_threshold_min = pickle.load(f)

    #with open('df_results.pkl', 'rb') as f:
    #    df_results = pickle.load(f)

    #get high level stats for this test
    y_true = list(df_results.dataset == 'fakes')
    y_pred = list(df_results.max_pred)
    precision_max = sklearn.metrics.precision_score(y_true, y_pred)
    recall_max = sklearn.metrics.recall_score(y_true, y_pred)

    y_pred = list(df_results.min_pred)
    precision_min = sklearn.metrics.precision_score(y_true, y_pred)
    recall_min = sklearn.metrics.recall_score(y_true, y_pred)

    y_pred = list(df_results.either_pred)
    precision_either = sklearn.metrics.precision_score(y_true, y_pred)
    recall_either = sklearn.metrics.recall_score(y_true, y_pred)

    print("Validate on Timit, the case of testing all values in a sentence")
    print("===== Max checks ====")
    print("Recall: ", recall_max)
    print("Precision: ", precision_max)
    print()
    print("===== Min checks ====")
    print("Recall: ", recall_min)
    print("Precision: ", precision_min)
    print()
    print("===== Either checks ====")
    print("Recall: ", recall_either)
    print("Precision: ", precision_either)
    print()



    """Operation #3 """
    ##find ideal set for TIMIT Dataset
    #print("Finding ideal set...")
    #df_org_ranges = get_optimal_threshold(df_test, df_org_ranges) 
    #
    ##vote on a per sentence basis similar to how hadi did it in the original
    ##paper to determine our effectiveness
    #df_oper3 = opt_test_sentence(df_org_ranges, df_eval)

    #y_true = list(df_oper3.ground_truth == 'fakes')
    #y_pred = list(df_oper3.prediction.values)
    #recall = sklearn.metrics.recall_score(y_true, y_pred)  
    #precision = sklearn.metrics.precision_score(y_true, y_pred)  
    #
    #print('Idealset operation on Timit Evaluation Set')
    #print('Recall: ', recall)
    #print('Precision: ', precision)
    #print()

    ##delete most of the timit so that we can free up space
    #pickle.dump(df_org_ranges, open('df_org_ranges.pkl', 'wb'))
    #pickle.dump(df_oper3, open('df_results.pkl', 'wb'))
    #del df_oper3
    #del df_eval
    #del df_test
    #del df_timit
    #pickle.dump(df_results, open('df_results.pkl', 'wb'))
    #del df_results

    """Operation #4"""
    print("Loading Lyrebird...")
    #Lyrebird
    #add dataset label
    df_lyrebird_true = load_data('lyrebird_true')
    df_lyrebird_true = process_df(df_lyrebird_true)
    df_lyrebird_fakes = load_data('lyrebird_fake')
    df_lyrebird_fakes = process_df(df_lyrebird_fakes)

    #add dataset label
    df_lyrebird_true['dataset'] = 'true'
    df_lyrebird_fakes['dataset'] = 'fakes'
    
    #into a single df for analysis
    df_lyrebird = pd.concat([df_lyrebird_true, df_lyrebird_fakes],
            ignore_index=True)
    df_lyrebird.reset_index(drop=True, inplace=True)

    print("Loading complete")

    #using ranges found for TIMIT, can we still detect Lyrebird and ASV_Spoof
    #with threshold, use validation threshold to get preformance of technique
    df_results = non_opt_test_sentences(df_org_ranges, df_lyrebird,\
            sentence_threshold_max, sentence_threshold_min, 
            sentence_threshold_either)

    #get high level stats for this test
    y_true = list(df_results.dataset == 'fakes')
    y_pred = list(df_results.max_pred)
    precision_max = sklearn.metrics.precision_score(y_true, y_pred)
    recall_max = sklearn.metrics.recall_score(y_true, y_pred)

    y_pred = list(df_results.min_pred)
    precision_min = sklearn.metrics.precision_score(y_true, y_pred)
    recall_min = sklearn.metrics.recall_score(y_true, y_pred)

    y_pred = list(df_results.either_pred)
    precision_either = sklearn.metrics.precision_score(y_true, y_pred)
    recall_either = sklearn.metrics.recall_score(y_true, y_pred)

    print("Validate on Lyrebird, the case of testing all values in a sentence")
    print("===== Max checks ====")
    print("Recall: ", recall_max)
    print("Precision: ", precision_max)
    print()
    print("===== Min checks ====")
    print("Recall: ", recall_min)
    print("Precision: ", precision_min)
    print()
    print("===== Either checks ====")
    print("Recall: ", recall_either)
    print("Precision: ", precision_either)
    print()



    print("Loading ASV Spoof...")
    #delete lyrebird 
    del df_lyrebird
    pickle.dump(df_results, open('df_lyrebird_results.pkl', 'wb'))
    del df_results
    
    #ASV_SPOOF (needs to load multiple mongo collections and then 
    #concatenate
    #add dataset label
    #df_asv_sets = ['asv_spoof_b1', 'asv_spoof_b2_100000',
    #        'asv_spoof_b2_100k_200k', 'asv_spoof_b2_200k_300k',
    #        'asv_spoof_b2_300k_400k', 'asv_spoof_b2_400k_500k', 
    #        'asv_spoof_b2_500k_600k', 'asv_spoof_b2_600k_700k', 
    #        'asv_spoof_b2_700k_800k', 'asv_spoof_b2__800k_end']

    #list_df_asv = []
    #for name in tqdm(df_asv_sets, position=0, leave=True, desc='ASV Spoof...'):
    #    list_df_asv.append(process_df(load_data(name)))

    #print('#1')
    ##into a single df for analysis
    #df_asv = pd.concat(list_df_asv, ignore_index=True)
    #df_asv.reset_index(drop=True, inplace=True)
 
    #print('#2')
    ##add dataset label
    #df_asv.loc['dataset'] = 'fakes'
    #df_asv.loc[df_asv.filepath.str.contains('bonafide'), 'dataset'] = 'true'

    #for each asv attack, run a validation pass

    print("Validate on ASV, the case of testing all values in a sentence")
    asv_pickles = ['df_asv_A07.pkl', 'df_asv_A08.pkl', 'df_asv_A09.pkl',
            'df_asv_A10.pkl', 'df_asv_A12.pkl', 'df_asv_A13.pkl',
            'df_asv_A14.pkl', 'df_asv_A15.pkl', 'df_asv_A17.pkl',
            'df_asv_A18.pkl', 'df_asv_A19.pkl']
    df_asv_bon = pickle.load(open('asv_data_files/df_asv_bon.pkl', 'rb'))
    df_asv_bon['dataset'] = 'true'

    for asv_curr in asv_pickles:
        #load data file
        df_asv_curr = pickle.load(open('asv_data_files/' + asv_curr, 'rb'))
        df_asv_curr['dataset'] = 'fakes'
        
        #combine with df_asv_bon
        df_asv_curr = pd.concat([df_asv_curr, df_asv_bon], ignore_index=True)
        df_asv_curr.reset_index(drop=True, inplace=True)

        #with threshold, use validation threshold to get preformance of techn
        df_results = non_opt_test_sentences(df_org_ranges, df_asv_curr,\
                sentence_threshold_max, sentence_threshold_min, 
                sentence_threshold_either)

        #get high level stats for this test
        y_true = list(df_results.dataset == 'fakes')
        y_pred = list(df_results.max_pred)
        precision_max = sklearn.metrics.precision_score(y_true, y_pred)
        recall_max = sklearn.metrics.recall_score(y_true, y_pred)

        y_pred = list(df_results.min_pred)
        precision_min = sklearn.metrics.precision_score(y_true, y_pred)
        recall_min = sklearn.metrics.recall_score(y_true, y_pred)

        y_pred = list(df_results.either_pred)
        precision_either = sklearn.metrics.precision_score(y_true, y_pred)
        recall_either = sklearn.metrics.recall_score(y_true, y_pred)

        print("RESULTS FOR --> ", asv_curr)
        print("===== Max checks ====")
        print("Recall: ", recall_max)
        print("Precision: ", precision_max)
        print()
        print("===== Min checks ====")
        print("Recall: ", recall_min)
        print("Precision: ", precision_min)
        print()
        print("===== Either checks ====")
        print("Recall: ", recall_either)
        print("Precision: ", precision_either)
        print()

        pickle.dump(df_results, open('df_asv.pkl', 'wb'))
   
if __name__ == '__main__':
    main()
