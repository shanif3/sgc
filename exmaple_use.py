import pandas as pd
from src.graph_update import *
import MIPMLP


def main():
    # taxonomy_levels; (list) specify all the taxonomy levels you are using
    taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    # shuffle; (bool) shuffle mode
    shuffle = False
    # folder; (String) specify a folder to save the output
    if shuffle:
        folder = '/home/shanif3/Dyamic_data/GDM-original/src/Results_shuffle'
    else:
        folder = '/home/shanif3/Dyamic_data/GDM-original/src/Results'
    # Loading dataset time points
    timeA = pd.read_csv(f"src/Data/T[A].csv", index_col=0)
    timeB = pd.read_csv(f"src/Data/T[B].csv", index_col=0)
    timeC = pd.read_csv(f"src/Data/T[C].csv", index_col=0)
    tag = pd.read_csv(f"src/Data/tag.csv", index_col=0)

    # Preprocess the data using MIPMLP
    process_timeA = MIPMLP.preprocess(timeA, taxnomy_group="mean", normalization='none')
    process_timeB = MIPMLP.preprocess(timeB, taxnomy_group="mean", normalization='none')
    process_timeC = MIPMLP.preprocess(timeC, taxnomy_group="mean", normalization='none')

    # threshold_p_value; (float) threshold p value
    threshold_p_value = 0.05
    # k_comb; (int) specify how many nodes will be in each sub graph, default k=4
    k_comb = 4

    run(process_timeA, process_timeB, process_timeC, taxonomy_levels, folder, tag, threshold_p_value, k_comb,shuffle)


if __name__ == '__main__':
    main()
