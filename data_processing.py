#!/usr/bin/env python
import pandas as pd
import numpy as np
from skbio.stats.composition import clr
import sys
PATH_metab_df = sys.argv[1] #### path to metabolomic profiles used in training
PATH_micro_df = sys.argv[2] #### path to microbiome profiles used in training
PATH_external_meta_df = sys.argv[3] #### path to metabolomic profiles used in testing/prediction
PATH_external_micro_df = sys.argv[4] #### path to metabolome profiles used in testing/prediction
PATH_metabolome_annotated = sys.argv[5] #### path to the annotation of metabolites in metabolomic profiles

## Load microbiome compositions and metabolomic profiles from the PRISM and NLIBD dataset
metab_df = pd.read_csv(PATH_metab_df, index_col=0)
micro_df = pd.read_csv(PATH_micro_df, index_col=0)

external_metab_df = pd.read_csv(PATH_external_meta_df, index_col=0)
external_micro_df = pd.read_csv(PATH_external_micro_df, index_col=0)

metabolome_annotated = pd.read_csv(PATH_metabolome_annotated, index_col=0)

samples = np.intersect1d(metab_df.columns.values, micro_df.columns.values)
num_samples = len(samples)

metab_df = metab_df[samples]
micro_df = micro_df[samples]

for c in micro_df.columns:
    micro_df[c] = pd.to_numeric(micro_df[c])
    
for c in metab_df.columns:
    metab_df[c] = pd.to_numeric(metab_df[c])
    
external_samples = np.intersect1d(external_metab_df.columns.values, external_micro_df.columns.values)
external_metab_df = external_metab_df[external_samples]
external_micro_df = external_micro_df[external_samples]

for c in external_micro_df.columns:
    external_micro_df[c] = pd.to_numeric(external_micro_df[c])

for c in external_metab_df.columns:
    external_metab_df[c] = pd.to_numeric(external_metab_df[c])
        
num_external_samples = len(external_samples)   


## Centered Log-Ratio DataFrames
metab_comp_df = pd.DataFrame(data=np.transpose(clr(metab_df.transpose() + 1)), 
                             index=metab_df.index, columns=metab_df.columns)

external_metab_comp_df = pd.DataFrame(data=np.transpose(clr(external_metab_df.transpose() + 1)), 
                                      index=external_metab_df.index, columns=external_metab_df.columns)
    

micro_comp_df = pd.DataFrame(data=np.transpose(clr(micro_df.transpose() + 1)), 
                             index=micro_df.index, columns=micro_df.columns)
external_micro_comp_df = pd.DataFrame(data=np.transpose(clr(external_micro_df.transpose() + 1)), 
                             index=external_micro_df.index, columns=external_micro_df.columns)

micro_comp_df = micro_comp_df.transpose()
metab_comp_df = metab_comp_df.transpose()
external_micro_comp_df = external_micro_comp_df.transpose()
external_metab_comp_df = external_metab_comp_df.transpose()
print(micro_comp_df.shape, metab_comp_df.shape, external_micro_comp_df.shape, external_metab_comp_df.shape)


## Use the PRISM as the training set and NLIBD as the test set
X_train = micro_comp_df.values
y_train = metab_comp_df.values
X_test = external_micro_comp_df.values
y_test = external_metab_comp_df.values


## Save the training and test data
np.savetxt("./processed_data/X_train.csv", X_train, delimiter=',')
np.savetxt("./processed_data/y_train.csv", y_train, delimiter=',')
np.savetxt("./processed_data/X_test.csv", X_test, delimiter=',')
np.savetxt("./processed_data/y_test.csv", y_test, delimiter=',')


## Save compound names
metabolome_raw = pd.read_csv(PATH_metab_df, index_col=0)
np.savetxt("./processed_data/compound_names.csv", 
           metabolome_annotated.reindex(metabolome_raw.index)['Compound Name'].values, delimiter='\t', fmt = '%s')



