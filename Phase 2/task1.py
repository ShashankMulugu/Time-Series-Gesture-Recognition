"""
    Description: The code reduces the terms to k-latent semantics 
    using PCA, SVD, LDA or NMF as inputted by the user and the corresponding 
    k-latent semantics are listed in the form of <word-score> pairs sorted
    in non-increasing order of scores.
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
#from scipy.spatial.distance import cosine
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

def score_computation(name,model,latent):
   
    pca_comp = []
    for i in range(k):
        scores = pd.Series(model.components_[i])
        sorted_scores = scores.abs().sort_values(ascending=False)
    
        final = []
        for sem, var in zip(latent[i], sorted_scores):
            fin = []
            fin = [sem,var]
            final.append(fin)
        
        pca_comp.append(final)
    
    tempdf = pd.DataFrame(pca_comp)
    tempdf.to_csv(file_dir+'/'+name+'_components_'+vector_choice+'.csv', sep = ",")
    
    labels = ['PRINCIPLE COMPONENT ' + str(x) + ':' for x in range(1, k + 1)]
    for label, pc in zip(labels, pca_comp):
        print(label)
        print(pc[:20])
        print('\n')
    

def display_topics(model, feature_names, no_top_words):
    
    res = []
    for topic_idx, topic in enumerate(model.components_):
#        print ("k = ", topic_idx)
#        print ([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        res.append([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
    
    return res

file_dir = input("Enter the file directory:")   
red_choice = input("Enter the reduction that you would like to perform \n 1.PCA 2.SVD 3.LDA 4.NMF : " )
vector_choice = input("Enter the vector model of your choice \n 1.TF 2.TFIDF : ")
STR = file_dir+'/all_'+vector_choice+'_gestures.csv'
df = pd.read_csv(file_dir+'/all_'+vector_choice+'_gestures.csv', index_col=0)

# Perform PCA
if(red_choice == 'PCA'):
    k = int(input("Enter the value of k: "))
    pca_list = list(df.columns.values)      
    dfT = df.T
    
    scaled_data = preprocessing.scale(df)
    pca = PCA(n_components = k)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
        
#    print("Total Variance Accounted for: ", (pca.explained_variance_ratio_* 100).sum())        
#    print("\n")
    print("Latent Semantic Term - Weight Pairs: " )
    print()

    lat_semantics_pca = display_topics(pca, pca_list, len(pca.components_[0]))
    score_computation('PCA',pca,lat_semantics_pca)

    new_df = pd.DataFrame(pca_data)
    new_df = new_df.T
    new_df.columns = dfT.columns
    new_df = new_df.T
    new_df.to_csv(file_dir+'/Reduced_'+vector_choice+'_'+red_choice+'.csv', sep = ",")


# Perform SVD
elif(red_choice == 'SVD'):
    k = int(input("Enter the value of k: "))
    svd_list = list(df.columns.values)
    dfT = df.T
    
    scaled_data = preprocessing.scale(df)      
    svd = TruncatedSVD(n_components=k)
    svd.fit(scaled_data)
    svd_data = svd.transform(scaled_data)
    
    print("Latent Semantic Term - Weight Pairs: " )
    print()

    lat_semantics_svd = display_topics(svd, svd_list, len(svd.components_[0]))
    score_computation('SVD',svd,lat_semantics_svd)
        
    new_df = pd.DataFrame(svd_data)
    new_df = new_df.T
    new_df.columns = dfT.columns
    new_df = new_df.T
    new_df.to_csv(file_dir+'/Reduced_'+vector_choice+'_'+red_choice+'.csv', sep = ",")
    
# Perform LDA
elif(red_choice == 'LDA'):
    k = int(input("Enter the value of k: "))      
    lda_list=list(df.columns.values)
    dfT = df.T
    
#    scaled_data = preprocessing.scale(df)
    lda = LatentDirichletAllocation(n_topics= k)
    lda.fit(df)
    lda_data = lda.transform(df)        

    print("Latent Semantic Term - Weight Pairs: " )
    print()

    lat_semantics_lda = display_topics(lda, lda_list, len(lda.components_[0]))
    score_computation('LDA',lda,lat_semantics_lda)
            
    new_df = pd.DataFrame(lda_data)
    new_df = new_df.T
    new_df.columns = dfT.columns
    new_df = new_df.T
    new_df.to_csv(file_dir+'/Reduced_'+vector_choice+'_'+red_choice+'.csv', sep = ",")
    
# Perform NMF
else:
    k = int(input("Enter the value of k: "))
    nmf_list = list(df.columns.values)
    dfT = df.T
    
#    scaled_data = preprocessing.scale(df)
    nmf = NMF(n_components = k, init='random', random_state=0) 
    nmf.fit(df)
    nmf_data = nmf.transform(df)
        
    print("Latent Semantic Term - Weight Pairs: " )
    print()

    lat_semantics_nmf = display_topics(nmf, nmf_list, len(nmf.components_[0]))
    score_computation('NMF',nmf,lat_semantics_nmf)
        
    new_df = pd.DataFrame(nmf_data)
    new_df = new_df.T
    new_df.columns = dfT.columns
    new_df = new_df.T
    new_df.to_csv(file_dir+'/Reduced_'+vector_choice+'_'+red_choice+'.csv', sep = ",")
    
