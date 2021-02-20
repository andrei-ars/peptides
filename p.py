import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('meta.tsv', sep = '\t', index_col = 'index')
label = pd.read_csv('data.tsv', sep ='\t', usecols=['index', 'label'], index_col = 'index')

#data = pd.read_csv('data.tsv', sep ='\t', index_col = 'index')
#meta = pd.read_csv('meta.tsv', sep = '\t', index_col = 'index')

df = df.merge(label, left_index=True, right_index=True)

# Note each compound has single label = True within File
desc = df.groupby(['Peptide', 'File']).agg({'label':sum}).describe()
print(desc)

# filter by LABEL & aggregate values from similar SAMPLE_TYPE by mean
fdf = (df[df.label]
       .pivot_table(index = ['Peptide', 'species', 'value'], columns = 'sample_type', values = 'ClusterMean', aggfunc = 'mean')
       .reset_index()
      )
print(fdf.head())

# find ratio of A to B. Since values in log scale simply take difference
fdf['logratio'] = fdf['A'] - fdf['B']

# calculate how far away from ground true the ratio is
fdf['deviation'] = fdf['logratio'] - fdf['value']
fdf.head()

# calculate summary for each SPECIES:
# 1. accuracy - absolute median deviation to the expected value
# 2. precision - standard deviation of ratios
stats = (fdf
         .groupby('species')
         .agg(
             accuracy = ('deviation', lambda x: np.abs(np.median(x))),
             precision = ('logratio', np.std)
         )
        )
stats.head()




#-------------

species = "YEAS8"
peptide = "EAAIEASTR"

species = "ECOLI"
peptide = "ILEIEGLPDLK"


meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "A")]
meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "A")].ClusterMean.median()
meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "B")]
meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "B")].ClusterMean.median()