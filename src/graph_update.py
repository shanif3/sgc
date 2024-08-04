import os
import pickle
import sys
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
from itertools import combinations, product, combinations_with_replacement
import math
import numpy as np
import pandas as pd
import samba
import statsmodels
from matplotlib import pyplot as plt
from scipy.stats.mstats import spearmanr
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as smt
from statsmodels.formula.api import ols
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.anova import AnovaRM, anova_lm
import MIPMLP
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_curve, roc_auc_score
import scipy.stats as statss
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import warnings
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import math

global taxonomy_levels
global folder


def create_dicts(processed):
    """
    Creating a dict for each time point, each dict contains 6 levels of taxonomy levels (except for kingdom) and each
    level contains its bacterias and its values.
    :param processed: tuple that contains all the processed data
    :param mode: 'SameT' or 'nextT', default set to 'SameT'כ
    :return: dict_time, processed_list
    """
    # number_of time; indicates how many times we have
    number_of_time = len(processed)
    processed_list = list(processed)

    # all_samples_dict_time; (list) contains all the samples values even if there are not in all-time points.
    all_samples_no_matter_time_dict_time = [defaultdict(list) for _ in range(number_of_time)]

    # Iterating over each processed data at each time point, for each sample (row)
    for all_samples_processed, all_samples_dict_time in tqdm(zip(processed_list, all_samples_no_matter_time_dict_time),
                                                             total=len(processed_list)):

        # all_time_processed.index; each index represent a sample
        for sample in all_samples_processed.index:
            row = all_samples_processed.loc[sample]
            try:
                row = row.to_frame().T
            except:
                c = 0

            # micro2matrix; for each sample (row) we will translate its microbiome values and its cladogram into matrix
            array_of_imgs_sampleT, bact_names_sampleT, ordered_df_sampleT = samba.micro2matrix(row, folder='',
                                                                                               save=False)

            # if statement; safety check, bacteria has 7 taxonomy levels, sometimes there is an empty first row (so 8
            # taxonomy levels)-we will delete it. '1:' is for skipping the first row if its empty
            if bact_names_sampleT.shape[0] == 8 and array_of_imgs_sampleT.shape[1] == 8:
                array_of_imgs_sampleT = array_of_imgs_sampleT[:, 1:, :]
                bact_names_sampleT = bact_names_sampleT[1:, :]

            # range(1,7); starting from row 1, because the first taxonomy level is 'kingdom'- very general.
            for row_level in range(1, 7):
                # row_names; contains all the bact names that are in taxonomy level row_level
                row_names = bact_names_sampleT[row_level, :]
                # array_of_imgs_row_level; contains the value of each bac on the row_level
                array_of_imgs_row_level = array_of_imgs_sampleT[:, row_level, :].flatten()

                # There are repeating bacteria names, so we will remove duplicates and we will get just a single value for each bac
                unique_names, unique_indices = np.unique(row_names, return_index=True)
                for name, indices in zip(unique_names, unique_indices):
                    # to ignore names like '0.0' or ''
                    if name[0].isalpha():
                        # all_samples_dict_time; (dict) contains for each bacteria name its values over all the samples.
                        if name not in all_samples_dict_time:
                            all_samples_dict_time[name] = []
                        all_samples_dict_time[name].append(array_of_imgs_row_level[indices])

    # with open('Diabi_train/all_samples_dict_time.pkl', 'wb') as file:
    #     pickle.dump(all_samples_dict_time, file)
    return all_samples_no_matter_time_dict_time


def correlation_link(level_num, list_dict_time, distribution_corr, threshold, mode='SameT',
                     parents=None, processed=None):
    """

    :param distribution_corr:
    :param processed:
    :param dict_level_num:
    :param mode:
    :param level_num:
    :param list_dict_time:
    :param parents:
    :return:
    """
    # dict_in_level: (list) filtered dict that contains just the bac that are in level_num
    dict_in_level = []
    for dict_time in list_dict_time:
        filtered_dict = {taxa: values for taxa, values in dict_time.items() if taxa.count(";") == level_num}
        dict_in_level.append(filtered_dict)

    if parents is not None:
        mutual_taxa = [key for key in dict_in_level[0].keys() if any(key.startswith(parent) for parent in parents)]
    elif parents is None:
        # dict_in_level already consist mutual taxa (based on create_dicts function). dict_in_level[0]- because all the dict has the same amount of bacteria
        mutual_taxa = dict_in_level[0].keys()

    # adding to all_levels in 'level' row the values of the taxa from each time point

    if mode == 'SameT':
        # 0.65 and 0.7
        pairs_to_return, distribution_corr = sameT(mutual_taxa, dict_in_level, threshold=threshold,
                                                   distribution_corr=distribution_corr)

    elif mode == 'NextT':
        if processed is not None:
            pairs_to_return, distribution_corr = nextT(mutual_taxa, dict_in_level, processed, threshold=threshold,
                                                       distribution_corr=distribution_corr)

    return pairs_to_return, distribution_corr


def sameT(mutual_taxa, dict_in_level, threshold, distribution_corr):
    dict_level_num = {}
    for name in mutual_taxa:
        # taxa_value; (list) contains the bac values along the time points
        taxa_value = []
        for dict_time in dict_in_level:
            taxa_value.extend(dict_time.get(name, []))

        # adding the taxa_value to all_levels at the current level with the specify name
        if name not in dict_level_num:
            dict_level_num[name] = []
        dict_level_num[name].extend(taxa_value)

        # all_levels[level]; contains all the bac names with their values along the time points

    names = list(dict_level_num.keys())
    pairs = combinations(names, 2)
    pairs_to_return = []
    total_combinations = math.comb(len(names), 2)

    with tqdm(total=total_combinations) as pbar:
        # loop over all pairs combinations for correlation check
        for name1, name2 in pairs:
            pbar.update(1)
            array1 = dict_level_num[name1]
            array2 = dict_level_num[name2]
            # using spearman correlation- to address a correlation
            corr, pval = spearmanr(array1, array2)
            distribution_corr.append(corr)

            # pairs_to_remove; contains pairs that below the threshold and pval is 0 or above 0.05
            if corr >= threshold and pval < 0.05:
                pairs_to_return.append((name1, name2))

    return pairs_to_return, distribution_corr


def nextT(mutual_taxa, dict_in_level, processed, threshold, distribution_corr):
    # will return [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]
    pairs = list(combinations_with_replacement(mutual_taxa, 2))
    pairs_to_return = []
    dict_t = {}
    dict_t_1 = {}

    # number_of_pairs_time; (int) number of time pairs, for example we have 4 time steps, so we have 3 pairs of times: (T1,T2)(T2,T3),(T3,T4)
    number_pairs_of_time = len(dict_in_level) - 1

    with tqdm(total=len(pairs)) as pbar:
        # Iterating over all the pairs, and checking for spearman correlation
        for name1, name2 in pairs:
            pbar.update(1)
            list_bac1_t = dict_t.get(name1, [])
            list_bac2_t_1 = dict_t_1.get(name2, [])
            list_bac2_t = dict_t.get(name2, [])
            list_bac1_t_1 = dict_t_1.get(name1, [])
            bac1_exist = False
            bac2_exist = False
            if list_bac1_t != [] and list_bac1_t_1 != []:
                bac1_exist = True

            if list_bac2_t != [] and list_bac2_t_1 != []:
                bac2_exist = True

            flag_first = True
            # checking if the lists are empty or they got the value from the dict.
            if not bac1_exist or not bac2_exist:
                for pair_time_t in range(number_pairs_of_time):
                    mutual_index_t = processed[pair_time_t].index.intersection(processed[pair_time_t + 1].index)
                    mutual_index_location_at_t = np.where(processed[pair_time_t].index.isin(mutual_index_t))[0]
                    mutual_index_location_at_t1 = np.where(processed[pair_time_t + 1].index.isin(mutual_index_t))[0]

                    # flag_first; (bool) used to indicates if we are in the first time step, if so i want to add the value just for list1_t. so i will compare between T1 T2.
                    # adding T1 to list_bac1_t and T2 to list_bac2_t_1
                    if not bac1_exist:
                        # adding name1 at time t (same) to list_bac1_t
                        list_bac1_t.extend([dict_in_level[pair_time_t][name1][i] for i in mutual_index_location_at_t])
                        # adding name1 at time t+1 (next) to list_bac1_t_1
                        list_bac1_t_1.extend(
                            [dict_in_level[pair_time_t + 1][name1][i] for i in mutual_index_location_at_t1])
                    if not bac2_exist:
                        list_bac2_t.extend([dict_in_level[pair_time_t][name2][i] for i in mutual_index_location_at_t])
                        list_bac2_t_1.extend(
                            [dict_in_level[pair_time_t + 1][name2][i] for i in mutual_index_location_at_t1])

            if name1 not in dict_t:
                dict_t[name1] = []
                dict_t[name1] = list_bac1_t
            if name2 not in dict_t:
                dict_t[name2] = []
                dict_t[name2] = list_bac2_t

            if name1 not in dict_t_1:
                dict_t_1[name1] = []
                dict_t_1[name1] = list_bac1_t_1
            if name2 not in dict_t_1:
                dict_t_1[name2] = []
                dict_t_1[name2] = list_bac2_t_1

            # Compute the Spearman correlation between the pairs
            # [T1 T2] between [T2 T3]
            corr, pval = spearmanr(list_bac1_t, list_bac2_t_1)
            distribution_corr.append(corr)
            # if corr < 0.5 or pval > 0.05:
            # pval == 0 or
            if corr >= threshold and pval < 0.05:
                pairs_to_return.append((name1, name2))

            # [T2 T3] between [T1 T2]
            corr, pval = spearmanr(list_bac2_t, list_bac1_t_1)
            distribution_corr.append(corr)
            # if corr < threshold or pval > 0.05:
            #     # or pval == 0
            #     pairs_to_remove.append((name1, name2))
            if corr >= threshold and pval < 0.05:
                pairs_to_return.append((name2, name1))

    # in order if we have (a,a) twice.
    pairs_to_return = set(pairs_to_return)

    return pairs_to_return, distribution_corr


# Function to convert a dictionary to a frozenset of key-value pairs- to pair
def dict_to_frozenset(d):
    return frozenset(d.items())


def graph(dict_time, processed, threshold):
    """
    Building the graph. The workflow is by starting from mode='SameT' to collect correlations between bac at the same
    time point. Then, we will check mode= 'NextT' to collect correlations between bac at the current time points to
    the next one.
    :param sameT_tuple: containing all_levels_sameT (list) and dict_same_time (list)
    :param nextT_tuple: containing all_levels_nextT (list) and dict_next_time (list)
    :return:
    """
    taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']

    # indicates if the level we are rn has a parents, for exmaple we started our algorithm with pylum level taxonomy, so it doesnt have parents, but for the level after it, pylum bacterias will be the parents for their children
    parents = None
    node_neighborhood = {}
    distribution_corr_same = []
    distribution_corr_next = []

    # starting from level 1 ( phylum) to level 6 (species)
    for level in range(1, 7):
        level_name = taxonomy_levels[level]

        print(f"{level_name} level is starting")

        # undirected pairs

        # mode = 'SameT'; checking for correlations between bac at the same time points. For example, if we have 3 time
        # points, we will check correlation between 2 bac values along time points -->
        # as the following [T1 T2 T3] vs [T1 T2 T3]

        pairs, distribution_corr_same_output = correlation_link(level, dict_time,
                                                                distribution_corr=distribution_corr_same,
                                                                parents=parents,
                                                                threshold=threshold, mode='SameT')
        distribution_corr_same.extend(distribution_corr_same_output)
        selected_bac_same_T = []
        for u, v in pairs:
            u_key = (u, "same", "same")
            v_key = (v, "same", "same")
            if u_key not in node_neighborhood:
                node_neighborhood[u_key] = []
            if v_key not in node_neighborhood:
                node_neighborhood[v_key] = []
            node_neighborhood[u_key].append(v_key)
            node_neighborhood[v_key].append(u_key)

            selected_bac_same_T.append(u_key)
            selected_bac_same_T.append(v_key)

        # mode = 'NextT'; checking for correlations between bac at the same time points to the next time points.
        # For example, if we have 3 time points, we will check correlation between 2 bac values along time points --> as
        # the following [T1 T2] vs [T2 T3]
        directed_pairs, distribution_corr_next_output = correlation_link(level, dict_time,
                                                                         distribution_corr=distribution_corr_next,
                                                                         threshold=threshold,
                                                                         mode='NextT', parents=parents,
                                                                         processed=processed)

        distribution_corr_next.extend(distribution_corr_next_output)
        selected_bac_next_T = []
        # format {u: {'time': 'same', 'level': 'same'}}
        ###### its directed edge, u has an impact to time t+1 on v
        # because i have (u,v) (v,u)- so i  will add it once
        for u, v in directed_pairs:
            u_key = (u, "same", "same")
            v_key = (v, "next", "same")
            if u_key not in node_neighborhood:
                node_neighborhood[u_key] = []
            node_neighborhood[u_key].append(v_key)
            # if v node( at time same) is not in neighborhood, we will add his next time node
            if v not in node_neighborhood:
                node_neighborhood[v_key] = []
            # u bac represent time- same, vs v bac represent time- next
            selected_bac_same_T.append(u_key)
            selected_bac_next_T.append(v_key)

        # taking the children from the sameT and the nextT and union them

        # make a connection between the parents and the children
        if parents is not None:
            for children_list, time_condition in [(selected_bac_same_T, 'same'), (selected_bac_next_T, 'next')]:
                for parent in parents:
                    for child in children_list:
                        time = 'same' if time_condition == 'same' else 'next'
                        # child[0] and parent[0] indicate the bac name
                        if child[0].startswith(parent[0]):
                            child_key = (child[0], time, 'child')
                            parent_key = (parent[0], time, 'parent')
                            # if statement to check if they key is already there, because we are checking about the names
                            # we can also have mode same and next for the same name, so we dont want duplicates keys
                            if child_key not in node_neighborhood[parent]:
                                # adding to the exist parent a connection with its child
                                node_neighborhood[parent].append(child_key)

                            if parent_key not in node_neighborhood[child]:
                                node_neighborhood[child].append(parent_key)

        # updating the current children to be parents to the next taxonomy level
        children = set(set(selected_bac_same_T) | set(selected_bac_next_T))
        parents = children
        print(f"{level_name} level is done")

    print(f"The graph has {len(node_neighborhood)} nodes")
    plt.hist(distribution_corr_same, bins=10, edgecolor='black')
    plt.title('Distribution of Spearman Correlations same')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig(f"{folder}.dist_same.png")
    plt.show()
    plt.hist(distribution_corr_next, bins=10, edgecolor='black')
    plt.title('Distribution of Spearman Correlations next')
    plt.yscale('log')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.savefig(f"{folder}/dist_next.png")
    plt.show()
    with open(f'{folder}/my_dict.pickle', 'wb') as f:
        pickle.dump(node_neighborhood, f)
    return node_neighborhood


def fix_comb_format(line):
    comb = line.replace("(", '').replace(")", '').replace("'", '').replace('\n', '').replace(" ", '').replace(
        "\t", '')
    comb = comb.split(',')
    return comb


def checking_person(node_neighborhood_dict, combinations_graph_nodes, over_time, processed_list, dict_time, tag,
                    mode='train', shuffle=False):
    """

    :param over_time:
    :param processed_list:
    :param dict_time:
    :return:
    """

    if shuffle:
        tag = tag_shuffle(over_time, tag)
    times_letter = [chr(65 + i) for i in range(len(processed_list))]  # Generates ['A', 'B', 'C', 'D', ...]

    locations = list(range(len(combinations_graph_nodes)))
    data_frame = pd.DataFrame(0, index=locations, columns=['sick', 'healthy', 'not_in_sick', 'not_in_healthy'])
    number_graph = 0
    # dict_num_comb_index_samples; (dict) {comb: {healthy: index1,index2...},{sick:index5,index6..},{not_in_sick: index11},{not_in_healthy: index12}}
    # for each sub graph we will have its sick and healthy and not in comb samples indexes.
    dict_num_comb_index_samples = defaultdict(lambda: defaultdict(list))

    for index, node_comb in enumerate(tqdm(combinations_graph_nodes)):

        specify_nodes = {}

        for number_node, node in enumerate(node_comb):
            node_name = node[0]
            if node not in specify_nodes:
                specify_nodes[node_name] = []
            # node[2]; time attribute of node
            specify_nodes[node_name].append(node[2])
            # if we have the same node at another time(same,next)
            time_to_add = set(n[2] for n in node_neighborhood_dict[node] if
                              n[0] is node_name)
            if len(time_to_add) > 0:
                specify_nodes[node_name].extend(list(time_to_add))

        for sample in over_time.index:
            tag_index = 'healthy' if tag.loc[sample][0] == 0 else 'sick'
            opposite_tag_index = 'not_in_healthy' if tag.loc[sample][0] == 0 else 'not_in_sick'

            # comb_results; (dict) where the key are letter and the value are 1/0, the bac exist at the same time point(1) or no.
            comb_results = {}
            # bits; get the sample's performances along time (for example: ('A',1) ('B',1) ('C',0) where 1 indicates exists in
            # this time letter, otherwise 0)
            bits = over_time.loc[sample]
            # combi_time; get just the times where the sample exists
            combi_time = "".join([times_letter[place] for place, i in enumerate(bits) if i == 1])

            for tuple_dict_letter in zip(times_letter, dict_time):
                # letter; dict time letter ( A for the first dict and so on..)
                # dict_letter_time; dict at letter time ( dict at time A (0) and so on..)
                letter, dict_letter_time = tuple_dict_letter
                if letter in combi_time:
                    # time_point_index; letter time index ( 0 for A and so on..)
                    time_point_index = times_letter.index(letter)
                    # sample_time_index; the location(index) where the sample located among the other indexes at the
                    # same itme(for example: sample= 'H01' located second among other at the same time)
                    sample_time_index = np.where(np.array(processed_list[time_point_index].index) == sample)[0][0]
                    # for each node in the combination i will check if the node exist at the same time(bigger than 0)
                    # in the current sample
                    comb_results[letter] = [1 if dict_letter_time[node_name_in_comb[0]][sample_time_index] > 0 else 0
                                            for
                                            node_name_in_comb in node_comb]

            # now comb_results contains for each node at each time if it exist(1), otherwise (0).
            # for example (3 time steps and 4 nodes) comb_results= {'A':[1,1,1,0], 'B':[1,1,0,1], 'C': [0,1,1,1]}

            # for each time in sample i will check if the sub graph is there, if so +1 to sub graph
            # Iterating over each time
            # time_index; represents 0 for letter 'A' and so on.
            # tune_letter; represents the comb_results keys (letters: 'A'..)
            for time_index, time_letter in enumerate(comb_results):
                # mult; (int) indicates if all the nodes are exists at the current time, 1 if exist otherwise 0.
                mult = 1
                time_letter_values = comb_results[time_letter]
                # Iterating over node_comb
                for node_index, node in enumerate(node_comb):
                    node_name = node[0]
                    # time_modes_list; (list) the times that node_name are at.
                    time_modes_list = specify_nodes[node_name]
                    # time_mode_list has mode same and next
                    if 'next' in time_modes_list:
                        # Checking if I have access to the next time step (meaning if I have next time step at the current time step)
                        if time_index + 1 <= len(comb_results) - 1:
                            # next_time_letter; (string) indicates the next time letter, for example we are right now at letter 'A' so hte next letter is 'B'
                            next_time_letter = times_letter[time_index + 1]
                            # node_value_next_time; (int) indicates if the node are exist at the next time, 1 for exists otherwise 0.
                            node_value_next_time = comb_results[next_time_letter][node_index]
                            mult *= node_value_next_time
                    if 'same' in time_modes_list:
                        node_values_same_time = time_letter_values[node_index]
                        mult *= node_values_same_time

                if mult == 1:
                    data_frame.loc[number_graph, tag_index] += 1
                    # saving to the comb dict an index of sample that has the comb, based on its tag (healthy, sick)
                    #  the index sample will appear in the dict the same amount of time that he has this comb
                    dict_num_comb_index_samples[index][tag_index].append(sample)
                else:
                    data_frame.loc[number_graph, opposite_tag_index] += 1
                    # if the index sample doesn't have the comb we will add it to 'not_in' section
                    dict_num_comb_index_samples[index][opposite_tag_index].append(sample)

        number_graph += 1

    # if mode == 'test':
    #     return check_people_test(dict_num_comb_index_samples)

    # indices_to_remove; (list) getting the indices of sub graph( comb) that doesnt appear in no one
    indices_to_remove = data_frame[(data_frame['sick'] == 0) & (data_frame['healthy'] == 0)].index
    # filtered_combinations_list; (list) all the combinations that appear at least in one time step
    filtered_combinations_list = [value for idx, value in enumerate(combinations_graph_nodes) if
                                  idx not in indices_to_remove]

    # filtered_num_com_index_samples; (dict) updating the dict to the relevant comb that appear at least in one time step
    filtered_num_com_index_samples = defaultdict(dict)
    new_key = 0

    if not indices_to_remove.empty:
        for old_key, value in dict_num_comb_index_samples.items():
            if old_key not in indices_to_remove:
                filtered_num_com_index_samples[new_key] = value
                new_key += 1

    # if the mode is train i want to delete the rows( combinations) that are without any healthy and sick people
    # if mode is test/ train_shuffle we dont want to remove these combination- because we are testing the combination from the train on the test
    if mode == 'train':
        data_frame = data_frame[~data_frame.index.isin(indices_to_remove)]

    # because we deleted rows, now we will reset the index
    data_frame = data_frame.reset_index(drop=True)

    data_frame.to_csv(f"{folder}/health_sick_counts_per_graph.csv")
    return data_frame, filtered_num_com_index_samples, filtered_combinations_list


# in order to get how many sick and healthy i have in total in all times all persons ( yes, there is duplicates)
def count_sick_health(over_time, tag):
    count = {"sick": 0, "healthy": 0}
    times = ['A', 'B', 'C']

    for i in over_time.index:
        tag_index = 'healthy' if tag.loc[i][0] == 0 else 'sick'
        bits = over_time.loc[i]
        combi_time = "".join([times[place] for place, i in enumerate(bits) if i == 1])
        count[tag_index] += len(combi_time)

    sick = count['sick']
    healthy = count['healthy']
    print(f"The data contains {sick} sick patients and {healthy} healthy patients. ")
    return sick, healthy


def check_people_test(dict_num_comb_index_samples):
    # save the intersection index in sick, healthy
    # Extract all 'sick' lists
    intersection_sick_counter, intersection_sick_list = get_intersection_list(dict_num_comb_index_samples, type='sick')
    intersection_healthy_counter, intersection_healthy_list = get_intersection_list(dict_num_comb_index_samples,
                                                                                    type='healthy')
    intersection_not_in_sick_counter, intersection_not_in_sick_list = get_intersection_list(dict_num_comb_index_samples,
                                                                                            type='not_in_sick')
    intersection_not_in_healthy_counter, intersection_not_in_healthy_list = get_intersection_list(
        dict_num_comb_index_samples, type='not_in_healthy')

    not_in_intersection_sick = get_not_intersection_list(dict_num_comb_index_samples, intersection_sick_counter,
                                                         type='sick')
    not_in_intersection_healthy = get_not_intersection_list(dict_num_comb_index_samples, intersection_healthy_counter,
                                                            type='healthy')

    c = 0
    data = {
        'sick': [len(intersection_sick_list)],
        'healthy': [len(intersection_healthy_list)],
        'not_in_sick': [len(not_in_intersection_sick) + len(intersection_not_in_sick_list)],
        'not_in_healthy': [len(not_in_intersection_healthy) + len(intersection_not_in_healthy_list)],
    }
    data_frame = pd.DataFrame(data)
    data_frame = data_frame[(data_frame['sick'] > 0) | (data_frame['healthy'] > 0)]

    data_dict = defaultdict(list)

    # Insert the lists into the defaultdict
    data_dict['sick'] = intersection_sick_list
    data_dict['healthy'] = intersection_healthy_list
    data_dict['not_in_sick'] = not_in_intersection_sick + intersection_not_in_sick_list
    data_dict['not_in_healthy'] = not_in_intersection_healthy + intersection_not_in_healthy_list

    return data_frame, data_dict


def get_intersection_list(dict_num_comb_index_samples, type: str):
    type_lists = [Counter(dict_num_comb_index_samples[key][type]) for key in dict_num_comb_index_samples if
                  type in dict_num_comb_index_samples[key]]

    # Reduce to find the intersection across all lists
    if len(type_lists) >= 1:
        intersection_counter = reduce(lambda x, y: x & y, type_lists)

        # Convert the intersection counter to a list with duplicates
        intersection_list = list(intersection_counter.elements())
    else:
        intersection_counter = Counter()
        intersection_list = []

    return intersection_counter, intersection_list


def get_not_intersection_list(dict_num_comb_index_samples, intersection_counter, type: str):
    # Find all elements not in the intersection
    non_intersection_sick_lists = []
    for key in dict_num_comb_index_samples:
        if type in dict_num_comb_index_samples[key]:
            # Elements in the original list but not in the intersection
            non_intersection = list(
                (Counter(dict_num_comb_index_samples[key][type]) - intersection_counter).elements())
            non_intersection_sick_lists.append(non_intersection)

    # Flatten the list of non-intersections
    non_intersection_flattened = [item for sublist in non_intersection_sick_lists for item in sublist]
    return non_intersection_flattened


# By a given combination we will check the comb values all over the data
def checking_the_sub_graph_over_all(node_neighborhood, over_time, processed, all_samples_dict_time, comb, tag):
    # for number_node, node in enumerate(comb):
    #     time = []
    #     for neighborhood in node_neighborhood[node]:
    #         if next(iter(neighborhood)) in comb and next(iter(neighborhood)) is not node:
    #
    #             time_n = list(node_neighborhood[node][np.where == next(iter(neighborhood))].values())[0]['time']
    #             time.append(time_n)
    #             if len(time) == 3:
    #                 break
    #     if 'next' in time and 'same' in time:
    #         time_n = 'both'
    #         time.append(time_n)

    times = [chr(65 + i) for i in range(len(processed))]
    rows_to_add = []
    for i in over_time.index:
        tag_index = 'healthy' if tag.loc[i][0] == 0 else 'sick'
        bits = over_time.loc[i]
        combi_time = "".join([times[place] for place, i in enumerate(bits) if i == 1])

        for time_letter in combi_time:

            time_point_index = times.index(time_letter)
            # index_sample_in_time; (int) the index of the sample at time point
            index_sample_in_time = np.where(np.array(processed[time_point_index].index) == i)[0][0]

            # comb_values; (list) getting all the values for each node in the comb at the same time point.
            comb_values = [all_samples_dict_time[time_point_index][node][index_sample_in_time] for node in comb]

            new_row = {'Patient': i,
                       'Time': time_letter,
                       'Tag': tag_index}

            # Add bacteria values to new_row dynamically
            for idx, bac in enumerate(comb):
                new_row[bac] = comb_values[idx]

            rows_to_add.append(new_row)

    df = pd.DataFrame(rows_to_add, columns=['Patient', 'Time', 'Tag'] + comb)
    df.to_csv(f"{folder}/all_patients_over_time_with_comb.csv")


# perform chi square test
def chi_square(health_sick_counts_per_graph_table, sick, healthy, p_value_threshold=0.05):
    """
    The chi-square statistic measures the magnitude of the difference between the observed and expected values.
     while the p-value indicates the probability that such a difference could arise by chance.
     So we want to take the minimum chi-square statistic.
    :param p_value_threshold:
    :return:
    """
    # SICK 117
    # HELATHY 532
    results = []
    # excepted_row; (list) The comb is in all the sick and healthy patients meanwhile there is no one without this
    # comb(0 patients)
    excepted_row = [sick, healthy, 0, 0]

    df = health_sick_counts_per_graph_table
    for index, row in df.iterrows():
        observed = row[['sick', 'healthy', 'not_in_sick', 'not_in_healthy']].values
        sick = observed[0]
        healthy = observed[1]
        not_in_sick = observed[2]
        not_in_healthy = observed[3]

        chi2_stat, p_val, _, _ = chi2_contingency([observed, excepted_row])
        # ignore p_val that are equals to 0 or bigger than the threshold
        if p_val > p_value_threshold:
            # or p_val == 0
            continue

        results.append((index, sick, healthy, not_in_sick, not_in_healthy, chi2_stat, p_val))

    # Add the results to the DataFrame
    results_df = pd.DataFrame(results, columns=['graph_num', 'sick', 'healthy', 'not_in_sick', 'not_in_healthy', 'chi2',
                                                'p_value'])

    # Apply FDR correction
    p_values = results_df['p_value'].values
    rejected, p_values_corrected, _, _ = smt.multipletests(p_values, alpha=p_value_threshold, method='fdr_bh')
    results_df['p_value_corrected'] = p_values_corrected

    # significant_row; the row that is the most similar to the excepted row --> with the smallest chi square statistic
    significant_row = results_df.loc[results_df['chi2'].idxmin()]
    graph_num = int(significant_row['graph_num'])
    sick = int(significant_row['sick'])
    healthy = int(significant_row['healthy'])
    not_in_sick = int(significant_row['not_in_sick'])
    not_in_healthy = int(significant_row['not_in_healthy'])
    chi2 = float(significant_row['chi2'])
    p_value = float(significant_row['p_value'])
    p_value_FDR = float(significant_row['p_value_corrected'])

    with open(f'{folder}/combi.txt', 'r') as f:
        lines = f.readlines()
        # comb files start counting rows from 1, and grap_num counting start from 0. so we will +1
        comb = lines[graph_num + 1]
        print(
            f"""Combination: {comb}\n Chi-square: {chi2}\n graph index:{graph_num}\n p-value before FDR: {p_value} \n p-val after FDR: {p_value_FDR}\n sick: {sick}\n healthy: {healthy}\n not_in_sick: {not_in_sick} \n not_in_healthy: {not_in_healthy} """)

    results_df.to_csv(f"{folder}/chi_square_test.csv")
    return comb


def combination_node(node_neighborhood, k=4):
    """

    :param k: number of nodes in sub graph
    :param folder: path to folder containing my_dict.pickle
    :return: list of valid node_time_same combinations
    """

    combi = []
    c = 0
    nodes = list(node_neighborhood.keys())  # Convert keys to list for Python 3 compatibility
    node_combinations = combinations(nodes, k)
    total_combinations = math.comb(len(nodes), k)
    with tqdm(total=total_combinations) as pbar:
        for node_comb in node_combinations:
            temp_k = k
            # Update progress bar
            pbar.update(1)

            # if we have node that is of time Next, meaning it doesnt have bac at time same that a connection, we will decrease the k to be k-1 because it will be harder to find match like this because it dircted edge
            # we decrease k by one just one time, doesnt matter how many next nodes i have, this is my threshold
            for node in node_comb:
                if 'next' in node:
                    temp_k -= 1
                    break

            # count_child_parent_connections: (int) indicates how many child-parent connections we have.
            # If we have more than 2, we will not use this combination- subgraph.
            count_child_parent_connections = sum(
                1 for node1, node2 in combinations(node_comb, 2) for tuple_node in node_neighborhood[node1] if
                node2[0] in tuple_node and (tuple_node[2] == 'child' or tuple_node[2] == 'parent')
            )
            if count_child_parent_connections > 2:
                continue

            # Check if each node in the combination is connected to at least one other node in the combination
            all_connected = True
            all = set()
            for node in node_comb:
                # checking if i have connection with other nodes that are not me
                neighbors_in_comb = [tuple(sorted((n[0], node[0]))) for n in node_neighborhood[node] if
                                     n[0] in {comb[0] for comb in node_comb} and n[0] != node[0]]
                if not neighbors_in_comb:
                    all_connected = False
                    break
                all.update(neighbors_in_comb)
            if all_connected and len(all) >= temp_k:
                c += 1
                combi.append(node_comb)

    print(f"Found {c} valid combinations.")
    # checking if we many sub-graph, combination as we need to continue learning on the graph
    if c < 300:
        return False
    # save to pickle
    with open(f'{folder}/combi.pickle', 'wb') as f:
        pickle.dump(combi, f)
    return combi


def check_where_id(processed_list, folder):
    """
   A function that creates a matrix of samples over times, where 1 indicates that the sample exists at this time, otherwise 0
    :param processed_list: list with all the processed csv, where each processed csv points to a time.
    :return: None, saves the matrix to a CSV file
    """
    # Getting all the samples id all over processed_list
    ids = [item for sub in [item.index for item in processed_list] for item in sub]
    # Converting the ids to string
    ids = [str(i) for i in ids]
    # Creating a dictionary that points to each time point dict, for example over_time = {'A': {}, 'B': {}, 'C': {}}
    over_time = {chr(65 + i): {} for i in range(len(processed_list))}

    for i in ids:
        for index_pro_list, pro_list in enumerate(processed_list):
            # if i is in pro_list.index we will put 1
            if i in pro_list.index:
                over_time[chr(65 + index_pro_list)][i] = 1

    over_time_df = pd.DataFrame(over_time)
    over_time_df.fillna(0, inplace=True)
    over_time_df.to_csv(f"{folder}/over_time.csv")

    return over_time_df


def plot_real_vs_shuffle(score_real, score_shuffle, folder, plot_name):
    # Order the values by size
    if plot_name == 'Chi_square_test':
        real_p_sorted = np.sort(score_real)[::-1]
        shuffle_p_sorted = np.sort(score_shuffle)[::-1]
    else:
        # plot the p value in -log10(p)
        real_p_sorted = -np.log10(np.sort(score_real)[::-1])
        shuffle_p_sorted = -np.log10(np.sort(score_shuffle)[::-1])

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(real_p_sorted, label='Real Data', marker='x')
    plt.plot(shuffle_p_sorted, label='Shuffle Data', marker='x')
    plt.xlabel('Ordered Index')
    plt.ylabel('-log10(p_val)')
    plt.title(f'Score of Real and Shuffle Data- {plot_name}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f"{folder}/{plot_name}_Score_real_vs_shuffle.png")


def tag_shuffle(over_time, tag):
    # Calculate the number of non-zero entries for each row
    lengths = over_time.sum(axis=1)
    # Group sample indices by their lengths
    length_groups = {}
    for idx, length in lengths.items():
        length_groups.setdefault(length, []).append(idx)

    for length, indices in length_groups.items():
        group_tags = tag.loc[indices, 'Tag'].tolist()
        np.random.shuffle(group_tags)
        tag.loc[indices, 'Tag'] = group_tags

    return tag


def calculate_chi_square_score(data_frame):
    x = data_frame['sick']
    y = data_frame['healthy']
    w = data_frame['not_in_sick']
    z = data_frame['not_in_healthy']
    total = x + y + w + z

    rho_1 = (x + y) / total
    rho_2 = (w + z) / total

    x_e = rho_1 * (x + w)
    y_e = rho_1 * (y + z)
    w_e = rho_2 * (x + w)
    z_e = rho_2 * (y + z)

    scores = ((x - x_e) ** 2 / x_e) + ((y - y_e) ** 2 / y_e) + ((w - w_e) ** 2 / w_e) + ((z - z_e) ** 2 / z_e)

    # Calculate the p-value for the chi-square statistic
    degrees_of_freedom = 1  # For a 2x2 contingency table, degrees of freedom is typically 1
    p_values = 1 - stats.chi2.cdf(scores, degrees_of_freedom)

    # [1] for getting the p values after bh
    # p_values_corrected = statsmodels.stats.multitest.fdrcorrection(p_values)[1]

    # Create a DataFrame to store graph numbers and scores
    score_df = pd.DataFrame({
        'graph_number': data_frame.index,
        'sick': data_frame['sick'],
        'healthy': data_frame['healthy'],
        'not_in_sick': data_frame['not_in_sick'],
        'not_in_healthy': data_frame['not_in_healthy'],
        'score': scores,
        'p_value': p_values,
        # 'p_value_corrected': p_values_corrected
    })

    return score_df


def main():
    taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    shuffle = True
    global folder
    if shuffle:
        folder = '/home/shanif3/Dyamic_data/GDM-original/src/Results_shuffle'
    else:
        folder = '/home/shanif3/Dyamic_data/GDM-original/src/Results'

    # score_real = calculate_chi_square_score(shuffle=False)
    # score_shuffle = calculate_chi_square_score(shuffle=True)
    # plot(score_real, score_shuffle)
    # Loading dataset time points
    timeA = pd.read_csv(f"Data/T[A].csv", index_col=0)
    timeB = pd.read_csv(f"Data/T[B].csv", index_col=0)
    timeC = pd.read_csv(f"Data/T[C].csv", index_col=0)
    tag = pd.read_csv(f"GDM/tag.csv", index_col=0)

    # Fixing format
    timeA = timeA.rename(columns={col: col.split('.')[0] for col in timeA.columns})
    timeB = timeB.rename(columns={col: col.split('.')[0] for col in timeB.columns})
    timeC = timeC.rename(columns={col: col.split('.')[0] for col in timeC.columns})

    threshold_p_value = 0.05
    k_comb = 4

    run(timeA, timeB, timeC, taxonomy_levels, folder, tag, threshold_p_value, k_comb, shuffle)


def run(timeA, timeB, timeC, taxonomy_levels_param, folder_param, tag, threshold_p_value, k_comb, shuffle):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Starting at :", current_time)

    global taxonomy_levels, folder
    taxonomy_levels = taxonomy_levels_param
    folder = folder_param
    processed = timeA, timeB, timeC

    if not os.path.exists(folder):
        os.mkdir(folder)

    samples_all_time_dict_time, all_samples_dict_time = create_dicts(processed)

    # checking the samples over time on all the samples ( including samples that are not along all the time points)
    over_time = check_where_id(processed)
    if shuffle:
        tag = tag_shuffle(over_time, tag)
    node_neighborhood_dict = graph(samples_all_time_dict_time)
    combinations_graph_nodes = combination_node(k_comb)
    #
    health_sick_counts_per_graph_table = checking_person(node_neighborhood_dict, combinations_graph_nodes, over_time,
                                                         processed, all_samples_dict_time, tag)
    sick, healthy = count_sick_health(over_time, tag)
    comb = chi_square(health_sick_counts_per_graph_table, sick, healthy, p_value_threshold=threshold_p_value)

    checking_the_sub_graph_over_all(node_neighborhood_dict, over_time, processed, all_samples_dict_time, comb, tag)

    # plot()
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("finished at :", current_time)


def evaluate(k):
    global folder
    global taxonomy_levels
    taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    global shuffle
    shuffle = False
    warnings.filterwarnings("ignore")
    folder = '/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109'
    tag = pd.read_csv(f"{folder}/tag.csv", index_col=0)
    # tag = tag[['Patient_ID', 'caesar']]
    # tag.columns = ['ID', 'Tag']

    tag = tag[['Patient_ID', 'mother_health_status']]
    tag.columns = ['ID', 'Tag']
    tag['Tag'] = tag['Tag'].map({'Ob': 1, 'Nw': 0})
    # Reset the index to remove the RUN column
    tag.reset_index(drop=True, inplace=True)

    # Set the ID column as the new index
    tag.set_index('ID', inplace=True)
    # Keep only the first occurrence of each index
    tag = tag.groupby(tag.index).first()
    folder_no_norm = 'no_normalization'
    folder_with_norm = 'with_normalization'

    # GDM
    # data no normalization
    # folder_no_norm = 'no_normalization'
    # T1_no_norm = pd.read_csv(f"{folder_no_norm}/T1.csv", index_col=0)
    # T2_no_norm = pd.read_csv(f"{folder_no_norm}/T2.csv", index_col=0)
    # T3_no_norm = pd.read_csv(f"{folder_no_norm}/T3.csv", index_col=0)
    #
    # # data with normalization
    # folder_with_norm = "with_normalization"
    # T1_with_norm = pd.read_csv(f"{folder_with_norm}/T1.csv", index_col=0)
    # T2_with_norm = pd.read_csv(f"{folder_with_norm}/T2.csv", index_col=0)
    # T3_with_norm = pd.read_csv(f"{folder_with_norm}/T3.csv", index_col=0)

    # Fixing format
    # timeA = timeA.rename(columns={col: col.split('.')[0] for col in timeA.columns})
    # timeB = timeB.rename(columns={col: col.split('.')[0] for col in timeB.columns})
    # timeC = timeC.rename(columns={col: col.split('.')[0] for col in timeC.columns})

    # probiotic ceasar 144
    # three_month_no_norm = pd.read_csv(f"{folder}/{folder_no_norm}/3_months.csv", index_col=0)
    # twelve_month_no_norm = pd.read_csv(f"{folder}/{folder_no_norm}/12_months.csv", index_col=0)
    # twenty_four_month_no_norm = pd.read_csv(f"{folder}/{folder_no_norm}/24_months.csv", index_col=0)
    # at_birth_no_norm = pd.read_csv(f"{folder}/{folder_no_norm}/At_Birth.csv", index_col=0)
    #
    # three_month_with_norm = pd.read_csv(f"{folder}/{folder_with_norm}/3_months.csv", index_col=0)
    # twelve_month_with_nrom = pd.read_csv(f"{folder}/{folder_with_norm}/12_months.csv", index_col=0)
    # twenty_four_month_with_norm = pd.read_csv(f"{folder}/{folder_with_norm}/24_months.csv", index_col=0)
    # at_birth_with_norm = pd.read_csv(f"{folder}/{folder_with_norm}/At_Birth.csv", index_col=0)

    # processed = three_month_no_norm, twelve_month_no_norm, twenty_four_month_no_norm, at_birth_no_norm
    # processed_mipmlp = three_month_with_norm, twelve_month_with_nrom, twenty_four_month_with_norm, at_birth_with_norm

    # #fat 109
    A_no_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/no_normalization/A.csv",
                            index_col=0)
    B_no_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/no_normalization/B.csv",
                            index_col=0)
    C_no_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/no_normalization/C.csv",
                            index_col=0)
    D_no_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/no_normalization/D.csv",
                            index_col=0)
    E_no_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/no_normalization/E.csv",
                            index_col=0)

    A_with_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/with_normalization/A.csv",
                              index_col=0)
    B_with_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/with_normalization/B.csv",
                              index_col=0)
    C_with_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/with_normalization/C.csv",
                              index_col=0)
    D_with_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/with_normalization/D.csv",
                              index_col=0)
    E_with_norm = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/with_normalization/E.csv",
                              index_col=0)

    processed = A_no_norm, B_no_norm, C_no_norm, D_no_norm, E_no_norm
    processed_mipmlp = A_with_norm, B_with_norm, C_with_norm, D_with_norm, E_with_norm

    checkpoint_flag = True
    if not checkpoint_flag:
        over_time = check_where_id(processed, folder)

        train_index, test_index = train_test_split(over_time.index, test_size=0.2)
        train_tag = tag.loc[train_index]
        test_tag = tag.loc[test_index]

        processed_train = []
        processed_test = []
        for index, proc_time in enumerate(processed):
            available_index_train = train_index.intersection(proc_time.index)
            available_index_test = test_index.intersection(proc_time.index)

            # groupby- in order to drop duplicates
            processed_train.append(proc_time.loc[available_index_train].groupby(level=0).first())
            processed_test.append(proc_time.loc[available_index_test].groupby(level=0).first())

        processed_train_aftermipmlp = []
        for proc_time_mip in processed_mipmlp:
            available_index_train = train_index.intersection(proc_time_mip.index)
            processed_train_aftermipmlp.append(proc_time_mip.loc[available_index_train].groupby(level=0).first())

        all_samples_dict_time_train = create_dicts(processed_train)
        all_samples_dict_time_train_Aftermipmlp = create_dicts(processed_train_aftermipmlp)
        all_samples_dict_time_test = create_dicts(processed_test)

        threshold = 0.6
        while True:
            node_neighborhood_train = graph(all_samples_dict_time_train, processed_train, threshold)
            # node_neighborhood_test = graph(samples_all_time_dict_time_test, processed_test, threshol)

            combinations_graph_train_nodes = combination_node(node_neighborhood_train, k=k)
            # if we dont have enough sub graph- we will decrease the threshold value for correlation edge in 20 percent
            if combinations_graph_train_nodes == False:
                threshold -= threshold * 0.2
            else:
                break

        checkpoint = {
            'train_index': train_index,
            'test_index': test_index,
            'train_tag': train_tag,
            'test_tag': test_tag,
            'processed_train': processed_train,
            'processed_test': processed_test,
            'processed_train_aftermipmlp': processed_train_aftermipmlp,
            'all_samples_dict_time_train': all_samples_dict_time_train,
            'all_samples_dict_time_train_Aftermipmlp': all_samples_dict_time_train_Aftermipmlp,
            'node_neighborhood_train': node_neighborhood_train,
            'combinations_graph_train_nodes': combinations_graph_train_nodes,
            'over_time': over_time
        }

        # Save the checkpoint using pickle
        with open(f'{folder}/checkpoint.pkl', 'wb') as f:
            pickle.dump(checkpoint, f)

    if checkpoint_flag:
        with open(f'{folder}/checkpoint.pkl', 'rb') as f:
            checkpoint = pickle.load(f)

        print("Loading checkpoint")
        # Access the saved variables
        train_index = checkpoint['train_index']
        test_index = checkpoint['test_index']
        train_tag = checkpoint['train_tag']
        test_tag = checkpoint['test_tag']
        processed_train = checkpoint['processed_train']
        processed_test = checkpoint['processed_test']
        processed_train_aftermipmlp = checkpoint['processed_train_aftermipmlp']

        all_samples_dict_time_train = checkpoint['all_samples_dict_time_train']
        all_samples_dict_time_train_Aftermipmlp = checkpoint['all_samples_dict_time_train_Aftermipmlp']
        node_neighborhood_train = checkpoint['node_neighborhood_train']
        combinations_graph_train_nodes = checkpoint['combinations_graph_train_nodes']
        over_time = checkpoint['over_time']

    # p_val_interaction_list_no_shuffle, p_val_tag_list_no_shuffle = preprocess_anova(combinations_graph_train_nodes,
    #                                                                                 train_tag,
    #                                                                                 node_neighborhood_train,
    #                                                                                 over_time.loc[train_index],
    #                                                                                 processed_train_aftermipmlp,
    #                                                                                 all_samples_dict_time_train_Aftermipmlp,
    #                                                                                 shuffle=False)
    #
    # p_val_interaction_list_shuffle, p_val_tag_list_shuffle = preprocess_anova(combinations_graph_train_nodes,
    #                                                                           train_tag,
    #                                                                           node_neighborhood_train,
    #                                                                           over_time.loc[train_index],
    #                                                                           processed_train_aftermipmlp,
    #                                                                           all_samples_dict_time_train_Aftermipmlp,
    #                                                                           shuffle=True)
    # plot_real_vs_shuffle(p_val_interaction_list_no_shuffle, p_val_interaction_list_shuffle, folder,
    #                      plot_name="interaction_tag_bac_ANOVA")
    # plot_real_vs_shuffle(p_val_tag_list_no_shuffle, p_val_tag_list_shuffle, folder, plot_name='tag_ANOVA')

    # Get training data
    train_data_frame, dict_num_comb_index_samples_train, filtered_combinations_graph_nodes = checking_person(
        node_neighborhood_train,
        combinations_graph_train_nodes,
        over_time.loc[train_index], processed_train,
        all_samples_dict_time_train,
        train_tag, mode = 'train')
    #
    # train_data_frame_shuffle, _, _ = checking_person(
    #     node_neighborhood_train,
    #     filtered_combinations_graph_nodes,
    #     over_time.loc[train_index], processed_train,
    #     all_samples_dict_time_train,
    #     train_tag, shuffle=True, mode='train_shuffle')
    #
    # all_samples_dict_time_test = create_dicts(processed_test)
    # test_data_frame, _, _ = checking_person(
    #     node_neighborhood_train,
    #     filtered_combinations_graph_nodes,
    #     over_time.loc[test_index],
    #     processed_test,
    #     all_samples_dict_time_test,
    #     test_tag, mode='test')
    # #
    processed_train = pd.concat(processed_train)
    processed_test = pd.concat(processed_test)
    tag_train = get_tag(processed_train, tag)
    tag_test = get_tag(processed_test, tag)
    #
    # combinations_df = pd.DataFrame(filtered_combinations_graph_nodes, columns=['bac1', 'bac2', 'bac3'])
    #
    # train_data_frame.to_csv(f"{folder}/train_data.csv")
    # test_data_frame.to_csv(f"{folder}/test_data.csv")
    # # Calculate chi-square scores_train for training data
    # scores_train = calculate_chi_square_score(train_data_frame)
    # scores_train_shuffle = calculate_chi_square_score(train_data_frame_shuffle)
    # scores_test = calculate_chi_square_score(test_data_frame)
    # scores_test.to_csv(f"{folder}/scores_test.csv")
    # scores_train.to_csv(f"{folder}/scores_train.csv")
    #
    # plot_real_vs_shuffle(scores_train['score'], scores_train_shuffle['score'], folder, plot_name="Chi_square_test")
    # plot_chi_square_train_vs_test(scores_train, scores_test)
    # # merge to each graph its combination
    # scores_train = pd.concat([scores_train, combinations_df], axis=1)
    #
    # scores_train.to_csv(f"{folder}/scores_train.csv")

    # Sort graphs by their scores_train in descending order and get top 10 most significant combinations
    scores_train = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/scores_train.csv", index_col=0)
    sorted_indices = np.argsort(scores_train['score'])[::-1]
    best_auc = 0
    top_k_comb = [filtered_combinations_graph_nodes[i] for i in sorted_indices]
    kf = StratifiedKFold(n_splits=10, shuffle=True)

    for index, comb in enumerate(top_k_comb):
        fixed_comb_list = add_taxonomy_levels_comb([(combi[0]) for combi in comb])
        processed_train_adding_comb_col = create_column_comb(processed_train, fixed_comb_list)

        train_data_frame = processed_train_adding_comb_col[fixed_comb_list]
        fold_scores = []
        fold_probabilities = []
        fold_true_labels = []
        for fold_number, idx_split in enumerate(kf.split(train_data_frame, tag_train)):
            train_idx, val_idx = idx_split
            X_train, X_val = train_data_frame.iloc[train_idx], train_data_frame.iloc[val_idx]
            y_train, y_val = tag_train.iloc[train_idx], tag_train.iloc[val_idx]

            X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
            X_val.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_val]

            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='error',
                max_depth=6,
                eta=0.3,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False
            )

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
            val_probabilities = model.predict_proba(X_val)[:, 1]
            val_predictions = (val_probabilities > 0.5).astype(int)
            fold_probabilities.extend(val_probabilities)
            fold_true_labels.extend(list(y_val['Tag']))

            fold_score = accuracy_score(y_val, val_predictions)
            print(f"Fold number {fold_number}: {fold_score}")
            fold_scores.append(fold_score)

        mean_fold_score = np.mean(fold_scores)
        print(f"Mean Accuracy for combination {comb}: {mean_fold_score}")

        if index < 10:
            plot_roc(folder, fold_true_labels, fold_probabilities, name=index)
        else:
            break
        fpr, tpr, _ = roc_curve(fold_true_labels, fold_probabilities)
        roc_auc = auc(fpr, tpr)
        if roc_auc > best_auc:
            best_auc = roc_auc
            best_comb = comb
            best_fold_true_labels = fold_true_labels
            best_fold_probabilities = fold_probabilities

    plot_roc(folder, best_fold_true_labels, best_fold_probabilities, name=f'best comb in index {index}')

    # Train final model on the full training data with the best combination
    fixed_comb_list = add_taxonomy_levels_comb([(combi[0]) for combi in best_comb])
    processed_train = create_column_comb(processed_train, fixed_comb_list)
    train_data_frame = processed_train[fixed_comb_list]
    train_data_frame.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in
                                train_data_frame.columns]

    model_final = XGBClassifier(
        objective='binary:logistic',
        eval_metric='error',
        max_depth=6,
        eta=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False
    )
    model_final.fit(train_data_frame, tag_train)

    # Make predictions for test data
    processed_test_final = create_column_comb(processed_test.copy(), fixed_comb_list)
    test_data_frame_final = processed_test_final[fixed_comb_list]

    test_data_frame_final.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in
                                     test_data_frame_final.columns]

    probabilities_final = model_final.predict_proba(test_data_frame_final)[:,
                          1]  # Get probabilities for the positive class
    predictions_final = (probabilities_final > 0.5).astype(int)

    # Evaluate the final model
    accuracy_final = accuracy_score(tag_test, predictions_final)
    print(f"Final Accuracy: {accuracy_final}")

    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(
        {'index': tag_test.index,
         'real_Tag': tag_test['Tag'],
         'Prediction': predictions_final,
         'Probability': probabilities_final})
    predictions_df.to_csv(f"{folder}/predictions.csv", index=False)

    # Plot ROC curve for the final model
    plot_roc(folder, list(tag_test['Tag']), probabilities_final, "final_model")


# learning each time every 5 batches
# fold_scores = []
# fold_probabilities = []
# fold_true_labels = []
# for i in range(5, len(top_k_comb) + 1, 5):
#     comb_batch = top_k_comb[:i]
#     combined_fixed_comb_list = []
#
#     for comb in comb_batch:
#         fixed_comb_list = add_taxonomy_levels_comb([(combi[0]) for combi in comb])
#         combined_fixed_comb_list.extend(fixed_comb_list)
#
#     combined_fixed_comb_list = list(set(combined_fixed_comb_list))
#     processed_train_adding_comb_col = create_column_comb(processed_train, combined_fixed_comb_list)
#     train_data_frame = processed_train_adding_comb_col[combined_fixed_comb_list]
#
#     for fold_number, (train_idx, val_idx) in enumerate(kf.split(train_data_frame, tag_train)):
#         X_train, X_val = train_data_frame.iloc[train_idx], train_data_frame.iloc[val_idx]
#         y_train, y_val = tag_train.iloc[train_idx], tag_train.iloc[val_idx]
#
#         X_train.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train.columns]
#         X_val.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_val]
#
#         model = XGBClassifier(
#             objective='binary:logistic',
#             eval_metric='error',
#             max_depth=6,
#             eta=0.3,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             use_label_encoder=False
#         )
#
#         model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
#         val_probabilities = model.predict_proba(X_val)[:, 1]
#         val_predictions = (val_probabilities > 0.5).astype(int)
#         fold_probabilities.extend(val_probabilities)
#         fold_true_labels.extend(list(y_val['Tag']))
#
#         fold_score = accuracy_score(y_val, val_predictions)
#         print(f"Fold number {fold_number}: {fold_score}")
#         fold_scores.append(fold_score)
#
#     # mean_fold_score = np.mean(fold_scores)
#     # print(f"Mean Accuracy for combination batch {i}: {mean_fold_score}")
#
# plot_roc(folder, fold_probabilities, fold_true_labels, f"train_roc_curve")
#
# # Make predictions for test data
# processed_test_final = create_column_comb(processed_test, combined_fixed_comb_list)
# test_data_frame_final = processed_test_final[combined_fixed_comb_list]
# test_data_frame_final.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in
#                                  test_data_frame_final.columns]
#
# probabilities_final = model.predict_proba(test_data_frame_final)[:, 1]
# predictions_final = (probabilities_final > 0.5).astype(int)
#
# # Evaluate the final model
# accuracy_final = accuracy_score(tag_test, predictions_final)
# print(f"Final Accuracy for combination batch {i}: {accuracy_final}")
#
# # Plot ROC for test data
# plot_roc(folder, probabilities_final, tag_test, f"test_comb_batch_{i}")
#

def plot_chi_square_train_vs_test(scores_train, scores_test):
    # Compute the sum of 'health' and 'sick' for each row in the training set

    # Calculate fraction_chi_square for train dataset
    scores_train['fraction_chi_square'] = 0
    scores_train.loc[scores_train['healthy'] > scores_train['sick'], 'fraction_chi_square'] = scores_train['score']
    scores_train.loc[scores_train['healthy'] < scores_train['sick'], 'fraction_chi_square'] = -scores_train['score']

    # Calculate fraction_chi_square for test dataset
    scores_test['fraction_chi_square'] = 0
    # scores_test can have nan values on 'score' when the combination doesn't appear in no one - we will ignore them
    # by replacing the nan with integer 2, this way we set p_Value to be 1 which is not significant.
    # scores_test['p_value'].fillna(2, inplace=True)

    scores_test.loc[scores_test['healthy'] > scores_test['sick'], 'fraction_chi_square'] = scores_test['score']
    scores_test.loc[scores_test['healthy'] < scores_test['sick'], 'fraction_chi_square'] = -scores_test['score']

    significance_threshold = 0.05
    scores_train['is_significant'] = scores_train['p_value'] < significance_threshold

    combined_df = scores_train[['fraction_chi_square', 'is_significant']].join(scores_test[['fraction_chi_square']],
                                                                               lsuffix='_train', rsuffix='_test')

    # Ensure all indices match
    combined_df = combined_df.dropna()
    plt.figure(figsize=(10, 6))

    # Plot significant points
    plt.figure(figsize=(10, 6))
    plt.scatter(
        combined_df.loc[combined_df['is_significant'], 'fraction_chi_square_train'],
        combined_df.loc[combined_df['is_significant'], 'fraction_chi_square_test'],
        color='red',  # Color for significant points
        label='Significant',
        alpha=0.5
    )

    # Plot non-significant points
    plt.scatter(
        combined_df.loc[~combined_df['is_significant'], 'fraction_chi_square_train'],
        combined_df.loc[~combined_df['is_significant'], 'fraction_chi_square_test'],
        color='blue',  # Color for non-significant points
        label='Non-Significant',
        alpha=0.5
    )
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
    # Plotting
    plt.xlabel('Train- Chi-Square')
    plt.ylabel('Test- Chi-Square')
    plt.title("Chi-Square: Train vs Test", fontsize=15, fontweight="bold")
    plt.grid(True)
    plt.savefig("scatter_chi_square_train_vs_test.png")
    plt.show()

    plt.close()


def plot_combined_histogram(df, mode):
    df['sum_health_sick'] = df['healthy'] + df['sick']
    df_sorted = df.sort_values(by='sum_health_sick')

    # Plot the sorted sum_health_sick values
    plt.figure(figsize=(10, 6))
    plt.plot(df_sorted['sum_health_sick'].values, marker='o', linestyle='-', color='b')
    plt.xlabel('Ordered Rows (by Sum)')
    plt.ylabel('Sum of Healthy and Sick')
    plt.title(f'{mode}')
    plt.grid(True)

    plt.savefig(f"count_people_{mode}.png")


def plot_roc(folder, true_labels, pred_probabilities, name):
    fpr, tpr, _ = roc_curve(true_labels, pred_probabilities)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{name}", fontsize=15, fontweight="bold")
    plt.legend(loc='lower right')
    plt.savefig(f'{folder}/roc_curve_{name}.png')
    plt.close()


def create_column_comb(processed, fixed_comb_list):
    # if the column is not in the processed (because the nodes its not a leaf- so micro2matrix created him), we will take all the sons below him and get a mean of them
    for fixed_node in fixed_comb_list:
        if fixed_node in processed.columns:
            continue
        else:
            fixed_node_trim = ";".join([part for part in fixed_node.split(';') if len(part) > 3])
            len_node_name = len(fixed_node_trim.split(';'))
            all_sons = []
            while not all_sons:
                all_sons = []
                for col in processed.columns:
                    len_col = len(([part for part in col.split(';') if len(part) > 3]))
                    if col.startswith(fixed_node_trim) and len_col == len_node_name + 1:
                        all_sons.append(processed[col])

                if all_sons:  # Found sons, calculate mean
                    all_sons_df = pd.concat(all_sons, axis=1)
                    processed[fixed_node] = all_sons_df.mean(axis=1)
                    break
                else:  # No sons found, derive from lower level
                    dict_all = {}
                    for col in processed.columns:
                        if col.startswith(fixed_node_trim):
                            len_bac = len([part for part in col.split(';') if len(part) > 3])
                            if len_bac not in dict_all:
                                dict_all[len_bac] = []
                            dict_all[len_bac].append(col)

                    sorted_dict = dict(sorted(dict_all.items()))
                    sorted_keys = list(sorted_dict.keys())

                    if sorted_keys:  # Check if there are keys in sorted_dict
                        first_key_cols = sorted_dict[sorted_keys[0]]
                        for col in first_key_cols:
                            all_sons.append(processed[col])

                        col_name = first_key_cols[0]
                        col_name = col_name.split(';')
                        len_col_name = len([part for part in col_name if len(part) > 3])
                        list_a = []
                        col_name_father = ";".join(col_name[:len_col_name - 1])
                        list_a.append(col_name_father)
                        col_name_father = add_taxonomy_levels_comb(list_a, mode='thereare')
                        col_name_father = col_name_father[0]
                        if all_sons:  # Check if all_sons is not empty
                            all_sons_df = pd.concat(all_sons, axis=1)
                            processed[col_name_father] = all_sons_df.mean(axis=1)
                            all_sons = []

    return processed


def preprocess_anova(combinations_graph_nodes, tag, node_neighborhood_dict, over_time, processed_list, dict_time,
                     shuffle=False):
    if shuffle:
        tag = tag_shuffle(over_time, tag)
    p_val_interaction_list = []
    p_val_tag_list = []
    times_letter = [chr(65 + i) for i in range(len(processed_list))]  # Generates ['A', 'B', 'C', 'D', ...]
    p_val = 0.05
    for index, node_comb in enumerate(tqdm(combinations_graph_nodes)):

        specify_nodes = {}

        for number_node, node in enumerate(node_comb):
            node_name = node[0]
            if node not in specify_nodes:
                specify_nodes[node_name] = []
            # node[2]; time attribute of node
            specify_nodes[node_name].append(node[2])
            # if we have the same node at another time(same,next)
            time_to_add = set(n[2] for n in node_neighborhood_dict[node] if
                              n[0] is node_name)
            if len(time_to_add) > 0:
                specify_nodes[node_name].extend(list(time_to_add))

        columns_bac_time = []
        for node_name in specify_nodes:
            # has only same and nextT
            if len(specify_nodes[node_name]) == 2:
                for time in specify_nodes[node_name]:
                    name_column = f"{node_name}_{time}"
                    columns_bac_time.append(name_column)
            if len(specify_nodes[node_name]) == 1:
                name_column = f"{node_name}_{specify_nodes[node_name][0]}"
                columns_bac_time.append(name_column)

        columns_bac_time.append('Tag')
        data_frame = pd.DataFrame(columns=columns_bac_time)

        for sample_index, sample in enumerate(over_time.index):

            tag_index = 'healthy' if tag.loc[sample][0] == 0 else 'sick'

            # comb_results; (dict) where the key are letter and the value are 1/0, the bac exist at the same time point(1) or no.
            comb_results = {}
            # bits; get the sample's performances along time (for example: ('A',1) ('B',1) ('C',0) where 1 indicates exists in
            # this time letter, otherwise 0)
            bits = over_time.loc[sample]
            # combi_time; get just the times where the sample exists
            combi_time = "".join([times_letter[place] for place, i in enumerate(bits) if i == 1])

            for tuple_dict_letter in zip(times_letter, dict_time):
                # letter; dict time letter ( A for the first dict and so on..)
                # dict_letter_time; dict at letter time ( dict at time A (0) and so on..)
                letter, dict_letter_time = tuple_dict_letter
                if letter in combi_time:
                    # time_point_index; letter time index ( 0 for A and so on..)
                    time_point_index = times_letter.index(letter)
                    # sample_time_index; the location(index) where the sample located among the other indexes at the
                    # same itme(for example: sample= 'H01' located second among other at the same time)
                    sample_time_index = np.where(np.array(processed_list[time_point_index].index) == sample)[0][0]
                    # for each node in the combination i will check if the node exist at the same time(bigger than 0)
                    # in the current sample
                    comb_results[letter] = [dict_letter_time[node_name_in_comb[0]][sample_time_index]
                                            for node_name_in_comb in node_comb]
            # row of sample, with the columns 'columns_bac_time', i will take from comb_results the values based on the columns time.
            # comb results eventually will contains matrix of number of time on k (number of bac) dim. si if i have a column that is next i will take the value for the next row in line as values.

            nodes_names_comb = [node[0] for node in node_comb]
            rows = []
            for time_index, letter in enumerate(comb_results):
                row = []
                for col in columns_bac_time:
                    if col == 'Tag':
                        row.append(tag_index)

                    else:

                        col_name_without_time, time_node = col.rsplit('_', 1)
                        index_between_other_nodes_comb_list = nodes_names_comb.index(col_name_without_time)
                        if time_node == 'next':
                            if time_index + 1 <= len(comb_results) - 1:
                                next_time_letter = times_letter[time_index + 1]
                                value_time = comb_results[next_time_letter][index_between_other_nodes_comb_list]
                                row.append(value_time)
                            else:
                                # we have node of next, but we dont have a row after us to take the value of the bac.
                                break
                        if time_node == 'same':
                            value_time = comb_results[letter][index_between_other_nodes_comb_list]
                            row.append(value_time)

                rows.append(row)
            data_frame = pd.concat([data_frame, pd.DataFrame(rows, columns=columns_bac_time)], ignore_index=True)

        p_value_interaction_tag_bac, p_value_tag = anova_repeated_measures(data_frame)
        p_val_interaction_list.append(p_value_interaction_tag_bac)
        p_val_tag_list.append(p_value_tag)
        # if p_value_interaction_tag_bac < p_val:
        #     best_comb = columns_bac_time
        #     p_val = p_value_interaction_tag_bac
    return p_val_interaction_list, p_val_tag_list


def anova_repeated_measures(data_frame):
    # Ensure 'Tag' column is of type 'category'
    data_frame['Tag'] = data_frame['Tag'].astype('category')
    data_frame = data_frame.reset_index()
    df_melt = pd.melt(data_frame, id_vars=['Tag'], value_vars=data_frame.columns,
                      var_name='Bacteria', value_name='Value')

    # Fit the ANOVA model
    model = ols('Value ~ C(Bacteria) + C(Tag) + C(Bacteria):C(Tag)', data=df_melt).fit()

    anova_table = anova_lm(model, typ=2)

    # Extract p-value for the interaction term
    p_value_interaction_tag_bac = anova_table.loc['C(Bacteria):C(Tag)', 'PR(>F)']
    p_value_tag = anova_table.loc['C(Tag)', 'PR(>F)']
    return p_value_interaction_tag_bac, p_value_tag


def create_comb_names(top_k_indices, filtered_combinations_graph_nodes):
    return [filtered_combinations_graph_nodes[idx] for idx in top_k_indices]


def convert_dict_graph_index_to_tag(test_top_k_graph_index_dict, tag):
    # merging all the default dict together, so we will get one dict for all.
    merged_dict_comb = defaultdict(list)
    # for comb_index in test_top_k_graph_index_dict:
    for key, value in test_top_k_graph_index_dict.items():
        merged_dict_comb[key].extend(value)

    predicted_tag_list_comb = []
    for key, values in merged_dict_comb.items():
        # len_values; (int) indicates how many values we have in this key
        len_values = len(values)
        # if key =='sick' we will put 1 as the length of the values
        if key == 'sick':
            predicted_tag_list_comb.extend([1] * len_values)
        # if key =='healthy' we will put 0 as the length of the values
        elif key == 'healthy':
            predicted_tag_list_comb.extend([0] * len_values)
        # if key =='not_in_sick' we will put 0 as the length of the values, because we didnt identify it as sick- so we will put the opposite tag
        elif key == "not_in_sick":
            predicted_tag_list_comb.extend([0] * len_values)
        # if key =='not_in_healthy' we will put 1 as the length of the values, because we didnt identify it as healthy- so we will put the opposite tag
        elif key == 'not_in_healthy':
            predicted_tag_list_comb.extend([1] * len_values)

    merged_list_all_indexes_by_order = []
    for key, value in merged_dict_comb.items():
        merged_list_all_indexes_by_order.extend(merged_dict_comb[key])
    real_tag_list_comb = [tag.loc[index][0] for index in merged_list_all_indexes_by_order]

    return real_tag_list_comb, predicted_tag_list_comb


def plot_auc(aucs, interval=5):
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(aucs) + 1), aucs, marker='o', linestyle='-', color='b', markersize=6, linewidth=2)
    plt.title('AUC Score vs. Top K Graphs', fontsize=16)
    plt.xlabel('Top K Graphs', fontsize=14)
    plt.ylabel('AUC Score', fontsize=14)
    plt.xticks(range(1, len(aucs) + 1, interval), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{folder}/auc_vs_top{len(aucs)}_comb.png", dpi=300)
    plt.show()


def get_tag(processed_train, tag):
    y_train = []
    for index in processed_train.index:
        y_train.append(tag.loc[index])

    data_frame_tag = pd.DataFrame(y_train)
    return data_frame_tag


def add_taxonomy_levels_comb(comb, mode='reg'):
    taxonomy = ['k__', ';p__', ';c__', ';o__', ';f__', ';g__', ';s__']
    all = []
    if mode == 'reg':
        for node_comb in comb:
            node_name = ''
            last_index = 0
            all_nodes = node_comb.split(';')
            for index, node in enumerate(all_nodes):
                node_name += taxonomy[index] + node
                last_index = index
            for index in range(last_index + 1, len(taxonomy_levels)):
                node_name += taxonomy[index]
            all.append(node_name)
    if mode == "thereare":
        for node_comb in comb:
            node_name = node_comb
            len_name_node = len(node_name.split(";"))
            for index in range(len_name_node, len(taxonomy_levels)):
                node_name += taxonomy[index]
            all.append(node_name)
    return all


if __name__ == '__main__':
    scores_train = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/scores_train.csv", index_col=0)
    scores_test = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/PRJNA1130109/scores_test.csv", index_col=0)
    #
    # scores_train['sum_health_sick'] = abs(scores_train['healthy'] - scores_train['sick'])/(scores_train['healthy'] + scores_train['sick'])

    # Identify rows in the training set where the sum is greater than 40
    # rmove = scores_train[scores_train['sum_health_sick'] < 0.8]

    # # Get the indices of these rows
    # filtered_indices = rmove.index
    #
    # # Remove the corresponding rows from the test set
    # scores_test = scores_test.drop(filtered_indices, errors='ignore')
    # scores_train=scores_train.drop(filtered_indices,errors='ignore')

    # p_values= scores_train['p_value']
    # rejected, p_values_corrected, _, _ = smt.multipletests(p_values, alpha=0.05, method='fdr_bh')
    # scores_train['p_value_corrected'] = p_values_corrected

    # plot_chi_square_train_vs_test(scores_train, scores_test)
    # plot_combined_histogram(scores_train, mode='train')
    # plot_combined_histogram(scores_test, mode='test')
    k = 3
    evaluate(k)


