# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 22:23:58 2016

@author: Zeyi
"""

import pandas as pd
import numpy as np
import scipy as sp
import random, string

def randomword(length):
   return ''.join(random.choice(string.lowercase) for i in range(length))

if __name__ == "__main__":
    
    max_url_sample = 1000
    num_entries  = 10000
    num_users = 100
    urllist = pd.read_csv('top-1m.csv', header=None, names=['rank','site'], usecols=['site'])
    urllist = urllist.head(max_url_sample) #Take only top 10K
    
    domains_idx = np.random.randint(0, max_url_sample, num_entries)
    domains_series = urllist.iloc[domains_idx].reset_index()
    domains_series.drop('index',inplace=True, axis=1)
    
    usernames = pd.Series([randomword(10) for i in range(num_users)]) #Random usernames
    user_idx = np.random.randint(0, num_users, num_entries)
    user_series = usernames[user_idx].reset_index()
    user_series.drop('index',inplace=True, axis=1)
    
    df = pd.concat([user_series, domains_series], axis=1)
    df.columns = ['user','domain']
    df_unique = df.groupby(['user','domain']).count().reset_index()

    
    domains = df_unique['domain'].unique()
    domain_dict = {x:i for i,x in enumerate(domains)}
    users = df_unique['user'].unique()
    users_dict = {x:i for i,x in enumerate(users)}  
    
    #Form normalized sparse adjacency graph of [users X domain]. We want column normalization
    # Find number of users in each domain. This allows normalization by column [Optional: can do this using scipy]
    domain_user_count = df_unique.groupby('domain')['user'].agg({'user_count':'nunique'})
    domain_user_count['value'] = 1.0/np.sqrt(domain_user_count['user_count'])
    domain_user_count['domain_idx'] = domain_user_count.index.map(lambda x: domain_dict[x])
    
    #Insert indices into df_unique
    df_unique['user_idx'] = df['user'].map(lambda x: users_dict[x])
    df_unique = df_unique.merge(domain_user_count, left_on='domain', right_index=True)
    
    sparse_X = sp.sparse.coo_matrix((df_unique['value'].values,(df_unique['user_idx'].values,df_unique['domain_idx'].values)),
                                     shape=(len(users),len(domains))) #COO matrix is good for instantiation
    sparse_X1 = sparse_X.tocsr()
    
    #Alternative for normalization here
    """
    norm = np.sqrt((sparse_X.power(2)).sum(axis=0))
    sparse_X = sparse_X.multiply(1.0/norm)
    """
    cov_X = (sparse_X.T).dot(sparse_X)
    #Now set diagonals to 0
    #cov_X.setdiag(0)
    
    #cov_X_thres = cov_X > 0.90 #Thresholding. The number can change to suit our tolerance
    
    masked_cov_X = cov_X.multiply(cov_X > 0.85)
    extracted_indices = sp.sparse.find(masked_cov_X) # Extract indices
    
    # Now we have a cocurrence count of each domain
    cocurrence_df = pd.concat((pd.Series(extracted_indices[i]) for i in range(3)), axis=1)
    cocurrence_df.columns = ['domain_idx','co_domain_idx','cocurrence_score']
    cocurrence_df['domain'] = cocurrence_df['domain_idx'].map(lambda x: domains[x])    
    cocurrence_df['co_domain'] = cocurrence_df['co_domain_idx'].map(lambda x: domains[x])   
    
    cocurrence_domains = cocurrence_df.groupby('domain').agg({'co_domain':['nunique','unique'], 'cocurrence_score':'max'})
    
    #Complete the matrix here
    
    #Inner join with something else
    
    
    