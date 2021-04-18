import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA

data_df = pd.read_csv("output\\all_tfidf_gestures.csv", index_col=0)


pca = PCA(n_components=len(data_df)//3, svd_solver='full')
red_df = pd.DataFrame(pca.fit_transform(data_df), index=data_df.index)
file_lst = list(red_df.index.values)
file_dct = {file_lst[i][:-4]: i for i in range(len(file_lst))}
red_df.reset_index(inplace=True)


def hash_func(vecs, projections):
    bools = np.dot(vecs, projections.T) > 0
    return [bool2int(str(bool_vec)) for bool_vec in bools]


def bool2int(x):
    y = 0
    for i, j in enumerate(x):
        if j:
            y += 1 << i
    return y


class Table:

    def __init__(self, hash_size, dim):
        self.table = dict()
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, dim)

    def add(self, vecs, label):
        entry = {'label': label}
        hashes = hash_func(vecs, self.projections)
        for h in hashes:
            if h in self.table:
                self.table[h].append(entry)
            else:
                self.table[h] = [entry]

    def query(self, vecs):
        hashes = hash_func(vecs, self.projections)
        results = list()
        for h in hashes:
            if h in self.table:
                results.extend(self.table[h])
        return results


class LSH:

    def __init__(self, num_tables, hash_size, dim):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.tables = list()
        for i in range(self.num_tables):
            self.tables.append(Table(self.hash_size, dim))

    def add(self, vecs, label):
        for table in self.tables:
            table.add(vecs, label)

    def query(self, vecs):
        results = list()
        for table in self.tables:
            results.extend(table.query(vecs))
        return results

    def describe(self):
        for table in self.tables:
            print(table.table)


class GestureSearch:

    def __init__(self, num_tables, hash_size, inp_dimensions, training_file):
        self.lsh = LSH(num_tables, hash_size, len(red_df.iloc[0])-1)
        self.training_file = training_file

    def train(self):
        for f in range(len(self.training_file)):
            self.lsh.add(
                self.training_file.iloc[f, 1:], self.training_file.loc[f, "index"][:-4])

    def query(self, query_file):
        results = self.lsh.query(query_file)
        print("Number of search results: " + str(len(results)))
        counts = dict()
        for r in results:
            if r['label'] in counts:
                counts[r['label']] += 1
            else:
                counts[r['label']] = 1
        for k in counts:
            counts[k] = float(counts[k])/len(red_df)
        return counts


if __name__ == "__main__":
    start_time = time.time()
    l = input("Enter number of layers (L): ")
    k = input("Enter number of hashes per layer (k): ")
    print("These are the files available in the input directory: \n")
    print(file_lst)
    q = input("Please pick a query file from the above directory: ")
    t = input("Enter number of similar files required (t): ")
    gs = GestureSearch(int(l), int(k), len(red_df.iloc[0])-1, red_df)
    gs.train()
    results = gs.query(red_df.iloc[file_dct[q], 1:])
    count = 0
    with open("output\\task3output.txt", "w") as f:
        for r in sorted(results, key=results.get, reverse=True):
            if count < int(t):
                print("<", r, ", ", results[r], ">", file=f)
                count += 1
    print("Number of gestures considered: ", len(red_df))
    print("Number of buckets considered: ", int(l)*int(k))
    print("Search Results:\n")
    count = 0
    for r in sorted(results, key=results.get, reverse=True):
        if count < int(t):
            print(str(count+1)+".", "<", r, ", ", results[r], ">")
            count += 1
        else:
            break
    end_time = time.time()
    print("\nRun Time: "+str(end_time-start_time)+" secs")
