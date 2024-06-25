import os
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import combinations, product
import math
import numpy as np
import pandas as pd
import samba
from matplotlib import pyplot as plt
from scipy.stats.mstats import spearmanr
from scipy.stats import chi2_contingency
import statsmodels.stats.multitest as smt
from tqdm import tqdm

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
        print("Something wrong with the data. ")
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


def correlation_link(dict_level_num, level_num, list_dict_time, parents, mode='SameT'):
    """

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
        pairs, pairs_to_remove = sameT(mutual_taxa, dict_level_num, dict_in_level, threshold=0.6)
    else:
        pairs, pairs_to_remove = nextT(mutual_taxa, dict_level_num, dict_in_level, threshold=0.4)

    for pair in pairs_to_remove:
        pairs.remove(pair)

    selected_bac_same_T = set([item for sublist in pairs for item in sublist])

    return pairs, selected_bac_same_T


def sameT(mutual_taxa, dict_level_num, dict_in_level, threshold):
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
        # distribution_corr.append(corr)

        # pairs_to_remove; contains pairs that below the threshold and pval is 0 or above 0.05
        if corr < threshold or pval == 0 or pval > 0.05:
            pairs_to_remove.append((name1, name2))

    return pairs, pairs_to_remove


def nextT(mutual_taxa, dict_level_num, dict_in_level, threshold):
    for name in mutual_taxa:
        value_t = []
        for dict_time_ in dict_in_level:
            value_t.extend(dict_time_.get(name, []))
        current_t = value_t[:-1]  # All elements except the last one
        next_t = value_t[1:]  # All elements except the first one
        if name not in dict_level_num:
            dict_level_num[name] = {"current_t": current_t, "next_t": next_t}

    names = set(dict_level_num.keys())
    # pairs = list(product(selected_bac_same_T, names))
    pairs = list(combinations(names, 2))
    pairs_to_remove = []

    for name1, name2 in pairs:
        array1_T_current_next = dict_level_num[name1]['current_t']
        array2_T_current_next = dict_level_num[name2]['next_t']

        # Compute the Spearman correlation between the pairs
        # [T1 T2] between [T2 T3]
        corr, pval = spearmanr(array1_T_current_next, array2_T_current_next)

        if corr < 0.5 or pval == 0 or pval > 0.05:
            pairs_to_remove.append((name1, name2))

        # [T2 T3] between [T1 T2]
        array1_T_current_next = dict_level_num[name2]['current_t']
        array2_T_current_next = dict_level_num[name1]['next_t']

        corr, pval = spearmanr(array1_T_current_next, array2_T_current_next)
        if corr < threshold or pval == 0 or pval > 0.05:
            pairs_to_remove.append((name1, name2))

    # in order if we have (a,a) twice.
    pairs_to_remove = set(pairs_to_remove)

    return pairs, pairs_to_remove


def graph(dict_time):
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

    # starting from level 1 ( phylum) to level 6 (species)
    for level in range(1, 7):
        level_name = taxonomy_levels[level]

        print(f"{level_name} level is starting")

        # undirected pairs

        # mode = 'SameT'; checking for correlations between bac at the same time points. For example, if we have 3 time
        # points, we will check correlation between 2 bac values along time points -->
        # as the following [T1 T2 T3] vs [T1 T2 T3]

        dict_level_num = {}
        pairs, selected_bac_same_T = correlation_link(dict_level_num, level, dict_time, parents, mode='SameT')

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
        directed_pairs, selected_bac_next_T = correlation_link(dict_level_num, level, dict_time, parents, mode='NextT')

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
        print(f"{level} level is done")

    print(f"The graph has {len(node_neighborhood)} nodes")
    with open(f'{folder}/my_dict.pickle', 'wb') as f:
        pickle.dump(node_neighborhood, f)


def fix_comb_format(line):
    comb = line.replace("(", '').replace(")", '').replace("'", '').replace('\n', '').replace(" ", '').replace(
        "\t", '')
    comb = comb.split(',')
    return comb


def checking_person(over_time, processed_list, dict_time, tag):
    """

    :param over_time:
    :param processed_list:
    :param dict_time:
    :return:
    """
    how_sick = 0
    how_health = 0
    with open(f'{folder}/my_dict.pickle', 'rb') as f:
        node_neighborhood = pickle.load(f)
    times_letter = [chr(65 + i) for i in range(len(processed_list))]  # Generates ['A', 'B', 'C', 'D', ...]

    with open(f"{folder}/combi.txt", 'r') as f:
        lines = f.readlines()
        locations = list(range(len(lines)))
        data_frame = pd.DataFrame(0, index=locations, columns=['sick', 'healthy', 'not_in_sick', 'not_in_healthy'])
        number_graph = 0

        for line in lines:
            comb = fix_comb_format(line)
            specify_node = []

            for number_node, node in enumerate(comb):
                time = []
                for neighborhood in node_neighborhood[node]:
                    if next(iter(neighborhood)) in comb and next(iter(neighborhood)) is not node:

                        time_n = list(node_neighborhood[node][np.where == next(iter(neighborhood))].values())[0]['time']
                        time.append(time_n)
                        # len(comb)-1; should be less than the comb len by 1, because its not including the node itself.
                        # so if the comb len is 4, i supposed to get maximum 3 items in time list
                        # ( why maximum? --> because not everyone is connecting to everyone)
                        if len(time) == len(comb) - 1:
                            break
                if 'next' in time and 'same' in time:
                    time_n = 'both'
                specify_node.append((number_node, node, time_n))

            for i in over_time.index:
                tag_index = 'healthy' if tag.loc[i][0] == 0 else 'sick'
                tag_index1 = 'not_in_healthy' if tag.loc[i][0] == 0 else 'not_in_sick'

                comb_results = {}
                bits = over_time.loc[i]
                combi_time = "".join([times_letter[place] for place, i in enumerate(bits) if i == 1])

                for index_time, tuple_dict_letter in enumerate(zip(times_letter, dict_time)):
                    letter, dict_letter_time = tuple_dict_letter
                    if letter in combi_time:
                        idx = np.where(np.array(processed_list[index_time].index) == i)[0][0]
                        # for each node in the combination i will check if it exist (bigger than 0) in the current sample
                        comb_results[letter] = [1 if dict_letter_time[node][idx] > 0 else 0 for node in comb]

                if tag_index == 'healthy':
                    how_health += len(comb_results)
                else:
                    how_sick += len(comb_results)
                # person_values = pd.concat([df for df in dfs])
                bacteria_next = [tup[0] for tup in specify_node if 'next' in tup]

                # how many times_letter i have
                # for each time in sample i will check if the sub graph is there, if so +1 to sub graph
                for i, time in enumerate(comb_results):
                    time = comb_results[time]
                    mult = 1
                    if len(bacteria_next) > 0:
                        # checking if i have an acseess to the next time if there is so
                        if i + 1 <= len(comb_results) - 1:
                            # multiple the next t
                            values_next = [comb_results[i + 1][node_next] for node_next in bacteria_next]
                            mult *= math.prod(values_next)
                        # if i dont have both ( same and next) and i have just next i dont want to multiple it in the same T so we will remove it
                        time = [i for index, i in enumerate(time) if index not in bacteria_next]
                    mult *= math.prod(time)
                    if mult == 1:
                        data_frame.loc[number_graph, tag_index] += 1
                    else:
                        data_frame.loc[number_graph, tag_index1] += 1
            number_graph += 1
        data_frame.to_csv(f"{folder}/health_sick_counts_per_graph.csv")
        return data_frame


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


# By a given combination we will check the comb values all over the data
def checking_the_sub_graph_over_all(over_time, processed, all_samples_dict_time, comb, tag):
    comb = fix_comb_format(comb)

    with open(f'{folder}/my_dict.pickle', 'rb') as f:
        node_neighborhood = pickle.load(f)

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

            time_point_index= times.index(time_letter)
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
        if p_val > p_value_threshold or p_val == 0:
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


def combination_node(k=4):
    """

    :param k: number of nodes in sub graph
    :param folder: path to folder containing my_dict.pickle
    :return: list of valid node combinations
    """
    # Load node neighbors from pickle file
    with open(f'{folder}/my_dict.pickle', 'rb') as f:
        node_neighbors = pickle.load(f)

    combi = []
    c = 0
    nodes = list(node_neighbors.keys())  # Convert keys to list for Python 3 compatibility
    node_combinations = list(combinations(nodes, k))

    with open(f"{folder}/combi.txt", 'w') as f, tqdm(total=len(node_combinations)) as pbar:
        for node_comb in node_combinations:
            # Update progress bar
            pbar.update(1)

            # count_child_parent_connections; (int) indicates how many child-parent connections we have.
            # If we have more than 2, we will not use this combination- sub graph.
            count_child_parent_connections = sum(
                1 for i, j in combinations(node_comb, 2) for dict_n in node_neighbors[i] if
                j in dict_n.keys() and dict_n[j]['level'] == 'child')
            if count_child_parent_connections > 2:
                continue

            # num_edges; (int) indicates how many connections we have among the nodes in such comb, we want more than
            # 4 connection in order to take it as sub graph.
            num_edges = sum(
                1 for i, j in combinations(node_comb, 2) for dict_n in node_neighbors[i] if j in dict_n.keys())

            if num_edges >= 4:
                c += 1
                combi.append(node_comb)
                f.write(f"{node_comb}\n")

    # Save valid combinations to pickle file
    with open(f'{folder}/combi.pickle', 'wb') as f:
        pickle.dump(combi, f)

    print(f"Found {c} valid combinations.")

    return combi


def check_where_id(processed_list):
    """
    A function that create a matrix of
    :param processed_list: list with all the processed csv, where each processed csv points to a time.
    :return: A matrix of samples on times, where 1 indicates that the sample is exists in this time, otherwise 0
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
    return over_time_df


def main():
    taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    folder = '/home/shanif3/Dyamic_data/GDM-original/src/Results'

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

    run(timeA, timeB, timeC, taxonomy_levels, folder, tag, threshold_p_value, k_comb)


def run(timeA, timeB, timeC, taxonomy_levels_param, folder_param, tag, threshold_p_value, k_comb):
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

    graph(samples_all_time_dict_time)
    combination_node(k_comb)

    health_sick_counts_per_graph_table = checking_person(over_time, processed, all_samples_dict_time, tag)
    sick, healthy = count_sick_health(over_time, tag)
    comb = chi_square(health_sick_counts_per_graph_table, sick, healthy, p_value_threshold=threshold_p_value)

    checking_the_sub_graph_over_all(over_time, processed, all_samples_dict_time, comb, tag)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("finished at :", current_time)


if __name__ == '__main__':
    main()

c = 0
