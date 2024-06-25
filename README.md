# **S**ub **G**raph **C**ount test
## Table of contents
- [Installation](https://github.com/shanif3/sgc#installation)
- [Project Structure](https://github.com/shanif3/sgc#project-structure)
- [Quick Tour](https://github.com/shanif3/sgc#quick-tour)
- [Example use](https://github.com/shanif3/sgc#example-use)
## Installation
``` bash
python pip sgc   
```
## Project structure 
In this repository, all the code and data are under `src` dir. You can find the `example_use`
``` bash
├── exmaple_use.py
├── README.md
├── src
│   ├── Data
│   │   ├── T[A].csv
│   │   ├── tag.csv
│   │   ├── T[B].csv
│   │   └── T[C].csv
│   ├── graph_update.py
│   └── Results
│       ├── all_patients_over_time_with_comb.csv
│       ├── chi_square_test.csv
│       ├── combi.txt
│       ├── health_sick_counts_per_graph.csv
│       └── my_dict.pickle

```

## Quick tour
In order to use our pacakge, please follow the tour.    
1. Preprocess your data using MIPMLP package, in order to do so please format the data as following:   
   - The first column is named "ID"
   - Each row represents a sample and each column represents an ASV.
   - The last row contains the taxonomy information, named "taxonomy".
2. After preprocess the data, Indicating you have T times points in the csv file data, divide the csv into T csvs. make sure you named the ids in each csv the same without any additional preffix or suffix for the time.
For example, in the attach dataset we 3 times points, the sample B013HN it appears in all of them. So insted of calling them B013HN_1 or B013HN_timeA for example, we will call them all with the same name- B013HN.
3. run `example_use.py` code

## Example use
Here we show how to use our package with an example GDM data given in our package.

```python
import pandas as pd
from src.graph_update import *
import MIPMLP


def main():
   # taxonomy_levels; (list) specify all the taxonomy levels you are using
   taxonomy_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
   # folder; (String) specify a folder to save the output
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

   run(process_timeA, process_timeB, process_timeC, taxonomy_levels, folder, tag, threshold_p_value, k_comb)

```


