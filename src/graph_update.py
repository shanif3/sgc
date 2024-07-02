import os
import pickle
import sys
from collections import defaultdict, Counter
from datetime import datetime
from functools import reduce
from itertools import combinations
import math
import numpy as np
import pandas as pd
import samba
from matplotlib import pyplot as plt
from scipy.stats.mstats import spearmanr
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as smt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import scipy.stats as statss

global taxonomy_levels
global folder


def create_dicts(processed):
    """
    Creating a dict for each time point, each dict contains 6 levels of taxonomy levels (except for kingdom) and each
    level contains its bacterias and its values.
    :param processed: tuple that contains all the processed data
    :param mode: 'SameT' or 'nextT', default set to 'SameT'
    :return: dict_time, processed_list
    """
    # number_of time; indicates how many times we have
    number_of_time = len(processed)
    processed_list = list(processed)

    # mutual_samples_over_time; (list) contains the samples that are all along the time points.
    mutual_samples_over_time = processed[0].index
    for i in range(1, number_of_time):
        mutual_samples_over_time = mutual_samples_over_time.intersection(processed[i].index)
    # processed_list; (list) processed with the mutual samples.
    samples_all_time_processed_list = [df.loc[mutual_samples_over_time] for df in processed]
    # all_same_shape; safety check that al the pr
    all_same_shape = all(
        df.shape[0] == samples_all_time_processed_list[0].shape[0] for df in samples_all_time_processed_list)
    # if statement; to check that all the processed datasets having the same shape or mutual_samples_over_time is
    # not empty
    if not all_same_shape or len(mutual_samples_over_time) == 0:
        sys.exit("Something wrong with the data. Existing the program")
        # TODO: handle here !

    # dict_time; (list) contains a dict for each time point.
    samples_all_time_dict_time = [defaultdict(list) for _ in range(number_of_time)]
    # all_samples_dict_time; (list) contains all the samples values even if there are not in all-time points.
    all_samples_no_matter_time_dict_time = [defaultdict(list) for _ in range(number_of_time)]

    # Iterating over each processed data at each time point, for each sample (row)
    for all_time_processed, all_time_dict_time, all_samples_processed, all_samples_dict_time in zip(
            samples_all_time_processed_list, samples_all_time_dict_time, processed_list,
            all_samples_no_matter_time_dict_time):

        # all_time_processed.index; each index represent a sample
        for sample in all_samples_processed.index:
            row = all_samples_processed.loc[sample]
            row = row.to_frame().T

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
                        # all_time_dict_time and all_samples_dict_time; will contain for each bac its value
                        # all_samples_dict_time include all_time_dict_time, since all_samples_dict_time is the
                        # processed as is and all_samples_dict_time is just the samples that are consist across all
                        # the time
                        if sample in all_time_processed.index:
                            if name not in all_time_dict_time:
                                all_time_dict_time[name] = []
                            all_time_dict_time[name].append(array_of_imgs_row_level[indices])
                        if name not in all_samples_dict_time:
                            all_samples_dict_time[name] = []
                        all_samples_dict_time[name].append(array_of_imgs_row_level[indices])

    return samples_all_time_dict_time, all_samples_no_matter_time_dict_time


def correlation_link(dict_level_num, level_num, list_dict_time, parents, distribution_corr, mode='SameT',
                     processed=None):
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

    # parents is not None; meaning that we have ancestor, and we want to take and continue building the graph based
    # on its children.
    # mutual_taxa; contains the mutual taxa along all the time points.
    if parents is not None:
        mutual_taxa = [key for key in dict_in_level[0].keys() if any(key.startswith(parent) for parent in parents)]
    elif parents is None:
        # dict_in_level already consist mutual taxa (based on create_dicts function)
        mutual_taxa = dict_in_level[0].keys()

    # adding to all_levels in 'level' row the values of the taxa from each time point

    if mode == 'SameT':
        pairs, pairs_to_remove, distribution_corr = sameT(mutual_taxa, dict_level_num, dict_in_level, threshold=0.65,
                                                          distribution_corr=distribution_corr)
    else:
        if processed is not None:
            pairs, pairs_to_remove, distribution_corr = nextT(mutual_taxa, dict_in_level, processed, threshold=0.7,
                                                              distribution_corr=distribution_corr)

    for pair in pairs_to_remove:
        pairs.remove(pair)

    selected_bac_same_T = set([item for sublist in pairs for item in sublist])

    return pairs, selected_bac_same_T, distribution_corr


def sameT(mutual_taxa, dict_level_num, dict_in_level, threshold, distribution_corr):
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
    pairs = list(combinations(names, 2))
    pairs_to_remove = []
    # loop over all pairs combinations for correlation check
    for name1, name2 in pairs:
        array1 = dict_level_num[name1]
        array2 = dict_level_num[name2]
        # using spearman correlation- to address a correlation
        corr, pval = spearmanr(array1, array2)
        distribution_corr.append(corr)

        # pairs_to_remove; contains pairs that below the threshold and pval is 0 or above 0.05
        if corr < threshold or pval > 0.05:
            # pval == 0 or
            pairs_to_remove.append((name1, name2))

    return pairs, pairs_to_remove, distribution_corr


def nextT(mutual_taxa, dict_in_level, processed, threshold, distribution_corr):
    # pairs = list(product(selected_bac_same_T, names))
    pairs = list(combinations(mutual_taxa, 2))
    pairs_to_remove = []
    dict_t = {}
    dict_t_1 = {}

    # number_of_pairs_time; (int) number of time pairs, for example we have 4 time steps, so we have 3 pairs of times: (T1,T2)(T2,T3),(T3,T4)
    number_pairs_of_time = len(dict_in_level) - 1
    # Iterating over all the pairs, and checking for spearman correlation
    for name1, name2 in pairs:
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
            for pair_time in range(number_pairs_of_time):
                mutual_index_t = processed[pair_time].index.intersection(processed[pair_time + 1].index)
                mutual_index_location = np.where(processed[pair_time].index.isin(mutual_index_t))[0]
                # flag_first; (bool) used to indicates if we are in the first time step, if so i want to add the value just for list1_t. so i will compare between T1 T2.
                if flag_first:
                    # adding T1 to list_bac1_t and T2 to list_bac2_t_1
                    if not bac1_exist:
                        list_bac1_t.extend([dict_in_level[pair_time][name1][i] for i in mutual_index_location])
                    if not bac2_exist:
                        list_bac2_t_1.extend([dict_in_level[pair_time][name2][i] for i in mutual_index_location])

                    # adding T1 to list_bac2_t and T2 to list_bac1_t_1
                    if not bac2_exist:
                        list_bac2_t.extend([dict_in_level[pair_time][name2][i] for i in mutual_index_location])
                    if not bac1_exist:
                        list_bac1_t_1.extend([dict_in_level[pair_time][name1][i] for i in mutual_index_location])
                    flag_first = False
                    continue

                if not bac1_exist:
                    list_bac1_t.extend([dict_in_level[pair_time][name1][i] for i in mutual_index_location])
                if not bac2_exist:
                    list_bac2_t_1.extend([dict_in_level[pair_time][name2][i] for i in mutual_index_location])

                if not bac2_exist:
                    list_bac2_t.extend([dict_in_level[pair_time][name2][i] for i in mutual_index_location])
                if not bac1_exist:
                    list_bac1_t_1.extend([dict_in_level[pair_time][name1][i] for i in mutual_index_location])

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
        if corr < 0.5 or pval > 0.05:
            # pval == 0 or
            pairs_to_remove.append((name1, name2))

        # [T2 T3] between [T1 T2]
        corr, pval = spearmanr(list_bac2_t, list_bac1_t_1)
        distribution_corr.append(corr)
        if corr < threshold or pval > 0.05:
            # or pval == 0
            pairs_to_remove.append((name1, name2))

    # in order if we have (a,a) twice.
    pairs_to_remove = set(pairs_to_remove)

    return pairs, pairs_to_remove, distribution_corr


def graph(dict_time, processed):
    """
    Building the graph. The workflow is by starting from mode='SameT' to collect correlations between bac at the same
    time point. Then, we will check mode= 'NextT' to collect correlations between bac at the current time points to
    the next one.
    :param sameT_tuple: containing all_levels_sameT (list) and dict_same_time (list)
    :param nextT_tuple: containing all_levels_nextT (list) and dict_next_time (list)
    :return:
    """

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

        dict_level_num = {}
        pairs, selected_bac_same_T, distribution_corr_same = correlation_link(dict_level_num, level, dict_time, parents,
                                                                              distribution_corr=distribution_corr_same,
                                                                              mode='SameT')

        for u, v in pairs:
            if u not in node_neighborhood:
                node_neighborhood[u] = []
            if v not in node_neighborhood:
                node_neighborhood[v] = []
            node_neighborhood[u].append({v: {'time': 'same', 'level': 'same'}})
            node_neighborhood[v].append({u: {'time': 'same', 'level': 'same'}})

        # mode = 'NextT'; checking for correlations between bac at the same time points to the next time points.
        # For example, if we have 3 time points, we will check correlation between 2 bac values along time points --> as
        # the following [T1 T2] vs [T2 T3]
        dict_level_num = {}
        directed_pairs, selected_bac_next_T, distribution_corr_next = correlation_link(dict_level_num, level, dict_time,
                                                                                       parents,
                                                                                       distribution_corr=distribution_corr_next,
                                                                                       mode='NextT',
                                                                                       processed=processed)

        ###### just for checking the sub graph im editing that the directed edges will be undircted
        for u, v in directed_pairs:
            if u not in node_neighborhood:
                node_neighborhood[u] = []
            if v not in node_neighborhood:
                node_neighborhood[v] = []
            node_neighborhood[u].append({v: {'time': 'next', 'level': 'same'}})
            node_neighborhood[v].append({u: {'time': 'next', 'level': 'same'}})

        # taking the children from the sameT and the nextT and union them
        children = set(selected_bac_same_T | selected_bac_next_T)

        # make a connection between the parents and the children
        if parents != None:
            for parent in parents:
                for child in children:
                    if child.startswith(parent):
                        node_neighborhood[parent].append({child: {'time': 'none', 'level': 'child'}})
                        node_neighborhood[child].append({parent: {'time': 'none', 'level': 'parent'}})

        parents = children
        print(f"{level_name} level is done")

    print(f"The graph has {len(node_neighborhood)} nodes")
    plt.hist(distribution_corr_same, bins=10, edgecolor='black')
    plt.title('Distribution of Spearman Correlations same')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.savefig(f"{folder}.dist_same.png")
    plt.show()
    plt.hist(distribution_corr_next, bins=10, edgecolor='black')
    plt.title('Distribution of Spearman Correlations next')
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
                    mode='train'):
    """

    :param over_time:
    :param processed_list:
    :param dict_time:
    :return:
    """

    times_letter = [chr(65 + i) for i in range(len(processed_list))]  # Generates ['A', 'B', 'C', 'D', ...]

    locations = list(range(len(combinations_graph_nodes)))
    data_frame = pd.DataFrame(0, index=locations, columns=['sick', 'healthy', 'not_in_sick', 'not_in_healthy'])
    number_graph = 0
    # dict_num_comb_index_samples; (dict) {comb: {healthy: index1,index2...},{sick:index5,index6..},{not_in_sick: index11},{not_in_healthy: index12}}
    # for each sub graph we will have its sick and healthy and not in comb samples indexes.
    dict_num_comb_index_samples = defaultdict(lambda: defaultdict(list))

    for index, comb in enumerate(tqdm(combinations_graph_nodes)):

        specify_node = []

        for number_node, node in enumerate(comb):
            time = []
            for neighborhood in node_neighborhood_dict[node]:
                if next(iter(neighborhood)) in comb and next(iter(neighborhood)) is not node:

                    time_n = list(node_neighborhood_dict[node][np.where == next(iter(neighborhood))].values())[0][
                        'time']
                    time.append(time_n)
                    # len(comb)-1; should be less than the comb len by 1, because its not including the node itself.
                    # so if the comb len is 4, i supposed to get maximum 3 items in time list
                    # ( why maximum? --> because not everyone is connecting to everyone)
                    if len(time) == len(comb) - 1:
                        break
            if 'next' in time and 'same' in time:
                time_n = 'both'
            if time != []:
                specify_node.append((number_node, node, time_n))

        for sample_index in over_time.index:
            tag_index = 'healthy' if tag.loc[sample_index][0] == 0 else 'sick'
            tag_index1 = 'not_in_healthy' if tag.loc[sample_index][0] == 0 else 'not_in_sick'

            comb_results = {}
            bits = over_time.loc[sample_index]
            combi_time = "".join([times_letter[place] for place, i in enumerate(bits) if i == 1])

            for tuple_dict_letter in zip(times_letter, dict_time):
                letter, dict_letter_time = tuple_dict_letter
                if letter in combi_time:
                    time_point_index = times_letter.index(letter)
                    idx = np.where(np.array(processed_list[time_point_index].index) == sample_index)[0][0]
                    # for each node in the combination i will check if it exist (bigger than 0) in the current
                    # sample
                    comb_results[letter] = [1 if dict_letter_time[node][idx] > 0 else 0 for node in comb]

            # bacteria_next; contains the index of the bacteria that have correlation between the current time to the next one.
            # tup[0] will return the index number of bacteria.
            bacteria_next = [tup[0] for tup in specify_node if 'next' in tup]

            # for each time in sample i will check if the sub graph is there, if so +1 to sub graph
            for i, time in enumerate(comb_results):
                time = comb_results[time]
                mult = 1
                # bacteria_next; contains the bacteria indexes that has correlation between the current time step to the next.
                # len(bacteria_next)>0; checking if there are indexes (bacteria) that have correlation to the next time step.
                if len(bacteria_next) > 0:
                    # Checking if I have access to the next time step (meaning if I have next time step at the current time step)
                    if i + 1 <= len(comb_results) - 1:
                        # multiply the next t:
                        # (list(comb_results.items()))[i+1] return the time_letter:values of k bacteria, at time i+1 (for example B:[1,1,,1,1] where k=4)
                        # (list(comb_results.items()))[i+1][1] return the values
                        # (list(comb_results.items()))[i+1][1][node_next_index] return the value at index node_next_index
                        values_next = [(list(comb_results.items()))[i + 1][1][node_next_index] for node_next_index in
                                       bacteria_next]
                        # multiply the next t
                        mult *= math.prod(values_next)
                    # time; (list) the values of the bacteria index that are not in
                    # bacteria_next, meaning that they have correlation at the same time step (sameT).
                    time = [i for index, i in enumerate(time) if index not in bacteria_next]
                mult *= math.prod(time)
                if mult == 1:
                    data_frame.loc[number_graph, tag_index] += 1
                    # saving to the comb dict an index of sample that has the comb, based on its tag (healthy, sick)
                    #  the index sample will appear in the dict the same amount of time that he has this comb
                    dict_num_comb_index_samples[index][tag_index].append(sample_index)
                else:
                    data_frame.loc[number_graph, tag_index1] += 1
                    # if the index sample doesn't have the comb we will add it to 'not_in' section
                    dict_num_comb_index_samples[index][tag_index1].append(sample_index)

        number_graph += 1

    if mode == 'test':
        return check_people_test(dict_num_comb_index_samples)

    # indices_to_remove; (list) getting the indices of sub graph( comb) that doesnt appear in no one
    indices_to_remove = data_frame[(data_frame['sick'] == 0) & (data_frame['healthy'] == 0)].index
    # filtered_combinations_list; (list) all the combinations that appear at least in one time step
    filtered_combinations_list = [value for idx, value in enumerate(combinations_graph_nodes) if
                                  idx not in indices_to_remove]

    # filtered_num_com_index_samples; (dict) updating the dict to the relevant comb that appear at least in one time step
    filtered_num_com_index_samples = defaultdict(dict)
    new_key = 0
    for old_key, value in dict_num_comb_index_samples.items():
        if old_key not in indices_to_remove:
            filtered_num_com_index_samples[new_key] = value
            new_key += 1

    # Remove graphs that appear in no one
    data_frame = data_frame[(data_frame['sick'] > 0) | (data_frame['healthy'] > 0)]
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
    for number_node, node in enumerate(comb):
        time = []
        for neighborhood in node_neighborhood[node]:
            if next(iter(neighborhood)) in comb and next(iter(neighborhood)) is not node:

                time_n = list(node_neighborhood[node][np.where == next(iter(neighborhood))].values())[0]['time']
                time.append(time_n)
                if len(time) == 3:
                    break
        if 'next' in time and 'same' in time:
            time_n = 'both'
            time.append(time_n)

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
    :return: list of valid node combinations
    """

    combi = []
    c = 0
    nodes = list(node_neighborhood.keys())  # Convert keys to list for Python 3 compatibility
    node_combinations = list(combinations(nodes, k))

    with tqdm(total=len(node_combinations)) as pbar:
        for node_comb in node_combinations:
            # Update progress bar
            pbar.update(1)

            # count_child_parent_connections; (int) indicates how many child-parent connections we have.
            # If we have more than 2, we will not use this combination- sub graph.
            count_child_parent_connections = sum(
                1 for i, j in combinations(node_comb, 2) for dict_n in node_neighborhood[i] if
                j in dict_n.keys() and dict_n[j]['level'] == 'child')
            if count_child_parent_connections > 2:
                continue

            # num_edges; (int) indicates how many connections we have among the nodes in such comb, we want more than
            # 4 connection in order to take it as sub graph.
            num_edges = sum(
                1 for i, j in combinations(node_comb, 2) for dict_n in node_neighborhood[i] if j in dict_n.keys())

            if num_edges >= 4:
                c += 1
                combi.append(node_comb)

    print(f"Found {c} valid combinations.")

    return combi


def check_where_id(processed_list):
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


def plot(score_real, score_shuffle):
    # Order the values by size
    real_p_sorted = np.sort(score_real)[::-1]
    shuffle_p_sorted = np.sort(score_shuffle)[::-1]

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(real_p_sorted, label='Real Data', marker='x')
    plt.plot(shuffle_p_sorted, label='Shuffle Data', marker='x')
    plt.xlabel('Ordered Index')
    plt.ylabel('Score')
    plt.title('Score of Real and Shuffle Data')
    plt.legend()
    plt.grid(True)
    plt.savefig("Score_real_vs_shuffle.png")


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

    # Create a DataFrame to store graph numbers and scores
    score_df = pd.DataFrame({
        'graph_number': data_frame.index,
        'sick': data_frame['sick'],
        'healthy': data_frame['healthy'],
        'not_in_sick': data_frame['not_in_sick'],
        'not_in_healthy': data_frame['not_in_healthy'],
        'score': scores
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
    tag = pd.read_csv(f"Data/tag.csv", index_col=0)

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


def evaluate(top_k_range):
    global folder
    global taxonomy_levels
    taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    global shuffle
    shuffle = False
    folder = '/home/shanif3/Dyamic_data/GDM-original/src/Diabi_train'
    # timeA = pd.read_csv(f"Data/T[A].csv", index_col=0)
    # timeB = pd.read_csv(f"Data/T[B].csv", index_col=0)
    # timeC = pd.read_csv(f"Data/T[C].csv", index_col=0)
    # tag = pd.read_csv(f"Data/tag.csv", index_col=0)
    #
    # # Fixing format
    # timeA = timeA.rename(columns={col: col.split('.')[0] for col in timeA.columns})
    # timeB = timeB.rename(columns={col: col.split('.')[0] for col in timeB.columns})
    # timeC = timeC.rename(columns={col: col.split('.')[0] for col in timeC.columns})

    timeA= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_1.csv",index_col=0)
    timeB= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_2.csv", index_col=0)
    timeC= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_3.csv", index_col=0)
    timeD= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_4.csv", index_col=0)
    timeE= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_5.csv", index_col=0)
    timeF= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_6.csv", index_col=0)
    timeG= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_7.csv", index_col=0)
    timeH= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_8.csv", index_col=0)
    timeI= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_9.csv", index_col=0)
    timeJ= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_10.csv", index_col=0)
    timeK= pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/diabimmune_check/collection_month_11.csv", index_col=0)
    tag= pd.read_csv("/home/shanif3/Codes/MIPMLP/data_to_compare/dynami/diabimmune/tag_filter.csv",index_col=0)

    # processed = timeA, timeB, timeC
    processed = timeA, timeB, timeC, timeD, timeE, timeF, timeG, timeH, timeI, timeJ, timeK

    over_time = check_where_id(processed)

    train_index, test_index = train_test_split(over_time.index, test_size=0.2)
    train_tag = tag.loc[train_index]
    test_tag = tag.loc[test_index]

    processed_train = []
    processed_test = []
    for index, proc_time in enumerate(processed):
        available_index_train = train_index.intersection(proc_time.index)
        available_index_test = test_index.intersection(proc_time.index)

        processed_train.append(proc_time.loc[available_index_train])
        processed_test.append(proc_time.loc[available_index_test])

    samples_all_time_dict_time_train, all_samples_dict_time_train = create_dicts(processed_train)
    samples_all_time_dict_time_test, all_samples_dict_time_test = create_dicts(processed_test)

    node_neighborhood_train = graph(all_samples_dict_time_train, processed_train)
    # node_neighborhood_test = graph(samples_all_time_dict_time_test)

    combinations_graph_train_nodes = combination_node(node_neighborhood_train, k=4)

    # Get training data
    train_data_frame, dict_num_comb_index_samples_train, filtered_combinations_graph_nodes = checking_person(
        node_neighborhood_train,
        combinations_graph_train_nodes,
        over_time.loc[train_index], processed_train,
        all_samples_dict_time_train,
        train_tag)

    combinations_df = pd.DataFrame(filtered_combinations_graph_nodes, columns=['bac1', 'bac2', 'bac3', 'bac4'])

    train_data_frame.to_csv(f"{folder}/train_data.csv")
    # Calculate chi-square scores for training data
    scores = calculate_chi_square_score(train_data_frame)
    # Ensure the sizes match
    # assert len(combinations_df) == len(train_data_frame), "Sizes do not match!"

    # merge to each graph its combination
    scores = pd.concat([scores, combinations_df], axis=1)

    scores.to_csv(f"{folder}/scores.csv")
    # Sort graphs by their scores in descending order
    sorted_indices = np.argsort(scores['score'])[::-1]

    aucs = []
    for k in range(1, top_k_range + 1):
        # Select top K graphs
        top_k_indices = sorted_indices[:k].values
        top_k_comb_name = create_comb_names(top_k_indices, filtered_combinations_graph_nodes)

        # when do model we will use this to take indexes of train top k
        # # Train classifier on training data using top K graphs
        # train_top_k_graph_data_frame = train_data_frame.iloc[top_k_indices]
        # train_top_k_graph_index_dict = [dict_num_comb_index_samples_train[key] for key in
        #                                 dict_num_comb_index_samples_train if key in top_k_indices]

        # Getting the test_data for the selected comb from train using top_k_comb_name i changed here to
        # node_neighborhood_train because there is times that the node_neighborhood doenst have the basteria because
        # it doesnt have correlation with other bacteria, so we will take the original, train dict and check if the
        # tag_index relvant to this comb using doct of the train
        test_data_frame, dict_num_comb_index_samples_test = checking_person(node_neighborhood_train,
                                                                            top_k_comb_name,
                                                                            over_time=over_time.loc[test_index],
                                                                            processed_list=processed_test,
                                                                            dict_time=all_samples_dict_time_test,
                                                                            tag=test_tag, mode='test')
        # if the comb didnt detect any sick or healthy samples, so i have no rows in the dataframe- auc is 0
        if test_data_frame.shape[0] == 0:
            aucs.append(0)

        else:
            real_tag_list_comb, predicted_tag_list_comb = convert_dict_graph_index_to_tag(
                dict_num_comb_index_samples_test,
                tag)

            auc = roc_auc_score(real_tag_list_comb, predicted_tag_list_comb)
            aucs.append(auc)

    plot_auc(aucs)


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


if __name__ == '__main__':
    # df = pd.read_csv("/home/shanif3/Dyamic_data/GDM-original/src/Results_train/scores.csv")
    # chi = df['score']
    # df['p_value'] = 1 - statss.chi2.sf(chi, 3)
    # df.to_csv("see.csv")
    evaluate(3)

c = 0
