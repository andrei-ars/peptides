import numpy as np
import pandas as pd
import logging
import pickle



def test():

    species = "YEAS8"
    peptide = "EAAIEASTR"

    species = "ECOLI"
    peptide = "ILEIEGLPDLK"

    meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "A")]
    meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "A")].ClusterMean.median()
    meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "B")]
    meta[(meta.species==species) & (meta.Peptide==peptide) & (meta.sample_type == "B")].ClusterMean.median()



class DataBase():

    def __init__(self, path):

        self.output_size = 12
        self.files_per_peptide = 6

        # Load the data
        # path = "data"
        self.data = pd.read_csv(path + '/data.tsv', sep ='\t', index_col = 'index')
        self.meta = pd.read_csv(path + '/meta.tsv', sep ='\t', index_col = 'index')
        self.meta['File'] = self.meta['File'].astype('int64', copy=False)
        #self.meta['index0'] = self.meta.index
        self.columns_number = len(self.data.columns)
        print("Data has been loaded")
        print("columns_number: {}".format(self.columns_number))

        with open(path + '/ndata.pickle', 'rb') as fp:
            self.ndata = pickle.load(fp)

        print("ndata has been loaded; ndata shape = {}".format(self.ndata.shape))
        print(self.ndata)

        #df = df.merge(label, left_index=True, right_index=True)
        self.peptides = self.meta.Peptide.unique() #36357 -- np.array
        #number_of_peptides = len(peptides)


    """
    def normalize_data(self, data):

        #for i, column in enumerate(data.columns):
        #    #column = 'MedianYI'
        #    #column = 'contain_bf'
        #    m = data[column].mean()
        #    std = data[column].std()
        #    func = lambda x: (x - m) / std
        #    data[str(i)] = data[column].apply(func)

        vdata = self.data.values # (821654, 32)  this creates a copy of data
        #np.apply_along_axis(function, 1, array)
        ndata = vdata.copy() # normalized data

        for j in range(vdata.shape[1]):
            print("processing {}-th column".format(j))
            m = vdata[:,j].mean()
            std = vdata[:,j].std()
            func = lambda x: (x - m) / std
            ndata[:,j] = np.apply_along_axis(func, 0, vdata[:,j])

        print("Normalized data:\n", ndata)
    """

    def get_meta_data(self, index):
        file_name = self.meta.loc[index, 'File']
        peptide_name = self.meta.loc[index, 'Peptide']
        species = self.meta.loc[index, 'species']
        cluster = self.meta.loc[index, 'Cluster']
        ClusterMean = self.meta.loc[index, 'ClusterMean']
        sample_type = self.meta.loc[index, 'sample_type']
        value = self.meta.loc[index, 'value']
        return {'ClusterMean': ClusterMean,
                'sample_type': sample_type,
                'value': value,
                }


    def get_data_for_peptide(self, peptide_number):
        pass

    def get_file_info(self, peptide_number, file_number):
        """ return np array 12 x 32
        first returns files from the group A, then from the group B
        """
        #print("file_number:", file_number)
        assert file_number < self.files_per_peptide
        peptide = self.peptides[peptide_number % len(self.peptides)]
        meta1 = self.meta[self.meta.Peptide==peptide]  # slow operation!
        files_A = meta1[(meta1.Peptide==peptide) & (meta1.sample_type=="A")].File.unique()
        files_B = meta1[(meta1.Peptide==peptide) & (meta1.sample_type=="B")].File.unique()
        files_A = list(files_A)
        files_B = list(files_B)
        #print("peptide: {}".format(peptide))
        #print("A:", files_A)
        #print("B:", files_B)
        if len(files_A) < 3:
            logging.warning("The list of files A is not full")
            print("peptide: {}".format(peptide))
            for _ in range(3 - len(files_A)):
                files_A.append(files_A[0])
        if len(files_B) < 3:
            logging.warning("The list of files B is not full")
            print("peptide: {}".format(peptide))
            for _ in range(3 - len(files_B)):
                files_B.append(files_B[0])

        if file_number < self.files_per_peptide//2:
            file = files_A[file_number]
            file_type = "A"
        else:
            file = files_B[file_number - self.files_per_peptide//2]
            file_type = "B"

        file_meta = meta1[(meta1.Peptide==peptide) & (meta1.File==file)]

        clusters = list(file_meta.Cluster)
        indices = list(file_meta.index)

        #print("clusters:", clusters)
        #print("cluster indices:", indices)
        #print(file_meta)

        new_indices = [indices[i % len(indices)]  for i in range(self.output_size)]

        #assert current_len <= m
        #if current_len < m:
            #lack = m - len(indices)
        #    for k in range(current_len, m):
        #        indices.append(indices[k % current_len])

        #print("new indices:", new_indices)
        array = self.ndata[new_indices]
        file = {'array': array, 'indices': new_indices}
        return file


    def analysis(self):
        df = pd.merge(meta, data, left_index=True, right_index=True)
        df1 = df[(df.sample_type == "A") & (df.value == 1)]
        df11 = df1[['MedianBSE', 'ClusterMean']]

    def f1():

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



if __name__ == "__main__":

    database = DataBase(path="data")
    #database.get_file_info(0, 0)
    file = database.get_file_info(1, 0)
    for i in file['indices']:
        meta_data = database.get_meta_data(i)
        print("i={}: {}".format(i, meta_data))

#-------------