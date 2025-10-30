#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 13:13:16 2020

@author: rst
"""


import numpy as np
import other_functions
import coeff_matrix
from sklearn import cluster


def k_SRBF(def_data,k_init,min_p,min_dist,workers):
    x_all = []
    y_all = []

    for i in range(len(def_data)):
        x_all = np.append(x_all,def_data[i][:,7])
        y_all = np.append(y_all,def_data[i][:,6])
    

    data_all = np.empty((len(x_all),2))
    data_all[:,0] = x_all
    data_all[:,1] = y_all
    kmeans = cluster.KMeans(n_clusters=k_init,n_init=1,max_iter=1000,tol=1e-10).fit(data_all)
    
    data_all = None
    C_clustered = kmeans.cluster_centers_
    x_rbf = C_clustered[:,0]
    y_rbf = C_clustered[:,1]
    idx = kmeans.labels_
##############################################################################    
    next_s = 1
    m = len(x_rbf)
    total_ind = [None]*m
    for i in range(m):
        total_ind[i] = np.where(idx == i)[0]
    
    indis1 = 0  
    while next_s == 1:
        for i in range(indis1,m):
            splt_cand = other_functions.dist(np.array([x_rbf[i]]),np.array([y_rbf[i]]),x_all[total_ind[i]],y_all[total_ind[i]],workers)
            
            if len(splt_cand[0,:])>=2*min_p: #len(np.where(splt_cand>=max_d)[1])>=1 and 
                indis1 = i
                break
            else:
                indis1 = i
            
        if indis1+1 == m and m>1:
            next_s = 0
        else:       
            data = np.empty((len(x_all[total_ind[indis1]]),2))
            data[:,0] = x_all[total_ind[indis1]]
            data[:,1] = y_all[total_ind[indis1]]
            kmeans = cluster.KMeans(n_clusters=2,max_iter=500,tol=1e-10).fit(data)
            C_clustered1 = kmeans.cluster_centers_
            
            x_rbf1 = C_clustered1[:,0]
            y_rbf1 = C_clustered1[:,1]
            idx_clustred1 = kmeans.labels_
            
            splt_test =  other_functions.dist(x_rbf1,y_rbf1,np.delete(x_rbf,indis1),np.delete(y_rbf,indis1),workers)

            splt_test2 = other_functions.dist(np.array([x_rbf1[0]]),np.array([y_rbf1[0]]),np.array([x_rbf1[1]]),np.array([y_rbf1[1]]),workers)
            
           
            if np.all(splt_test>min_dist) and np.all(splt_test2>min_dist) and len(np.where(idx_clustred1==0)[0])>min_p and len(np.where(idx_clustred1==1)[0])>min_p:
                x_rbf = np.append(x_rbf,x_rbf1)
                y_rbf = np.append(y_rbf,y_rbf1)
                
                
                total_ind.append(total_ind[indis1][np.where(idx_clustred1 == 0)[0]])
                total_ind.append(total_ind[indis1][np.where(idx_clustred1 == 1)[0]])
                
                x_rbf = np.delete(x_rbf,indis1)
                y_rbf = np.delete(y_rbf,indis1)

                total_ind.pop(indis1)
                m += 1
            else:
                indis1 += 1
##############################################################################  
    return(x_rbf,y_rbf)


def shape_sel_with_GCV(def_data,x_rbf,y_rbf,designed_coeff,designed_obs,workers,shape_range,total_data,function):
    
    print('\nOptimal depth selection started...')
    k = 0
    data_number = len(def_data)
    coeff_mat = [None]*data_number
    gcv = np.zeros(len(shape_range))
    u = np.random.binomial(1,0.5,total_data)
    u[u==0] = -1
    for j in shape_range:
       
        for i in range(data_number):
            coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(def_data[i][:,7],def_data[i][:,6],x_rbf,y_rbf,np.ones(len(x_rbf))*j,workers,designed_coeff[i],function)
        #print('\nParameter Estimation started. Please wait...')

        parameter,sigma = other_functions.MCVCE(coeff_mat,designed_obs)
        
        all_coeff = np.concatenate((coeff_mat[:]))
        coeff_mat = [None]*data_number
        all_res = np.concatenate((designed_obs[:]))
        
        
        all_coeff2 = np.copy(all_coeff)
        weight = np.ones(total_data)
        indis = 0
        for i in range(data_number):
            all_coeff2[indis:indis+len(designed_obs[i]),:] *= 1/(sigma[i][-1]**2)
            weight[indis:indis+len(designed_obs[i])] *= 1/(sigma[i][-1]**2)
            indis += len(designed_obs[i])
        Beta = np.linalg.solve((np.dot(all_coeff2.T,all_coeff)+(sigma[0][-1]**2/sigma[-1][-1]**2)*np.eye(len(parameter))),np.dot(all_coeff2.T,u))
        trace = np.dot(u.T,np.dot(all_coeff,Beta))
        gcv[k] = (total_data*np.sum((((np.dot(all_coeff,parameter)-all_res)*1e-5)**2)*weight)) / ((total_data-trace)**2)
        all_coeff = None
        all_coeff2 = None
        weight = None

        print('{:.1f} km depth is tested, GCV is {:.6e}'.format(j,gcv[k]))
        k += 1
    
    shape_last = shape_range[np.argmin(gcv)]
    # res_last = res[np.argmin(gcv)]
    return(shape_last,gcv)

def shape_sel_with_GCV_ind(x,y,x_rbf,y_rbf,designed_coeff,designed_obs,workers,shape_range,total_data,function):
    
    print('\nOptimal depth selection started...')
    k = 0
    data_number = len(x)
    coeff_mat = [None]*data_number
    gcv = np.zeros(len(shape_range))
    res = [None]*len(shape_range)
    u = np.random.binomial(1,0.5,total_data)
    u[u==0] = -1
    for j in shape_range:
       
        for i in range(data_number):
            coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(x[i],y[i],x_rbf,y_rbf,np.array([j]),workers,designed_coeff[i],function)
        #print('\nParameter Estimation started. Please wait...')

        parameter = other_functions.LS(coeff_mat,designed_obs)
        
        
        all_coeff = np.concatenate((coeff_mat[:]))
        coeff_mat = [None]*data_number
        all_res = np.concatenate((designed_obs[:]))
        
        Beta = np.linalg.solve(np.dot(all_coeff.T,all_coeff),np.dot(all_coeff.T,u))
        trace = np.dot(u.T,np.dot(all_coeff,Beta))
        gcv[k] = (total_data*np.sum(((np.dot(all_coeff,parameter)-all_res)*1e-5)**2)) / ((total_data-trace)**2)
        res[k] = np.dot(all_coeff,parameter)

        #print('{:.1f} km depth is tested, GCV is {:.6e}'.format(j,gcv[k]))
        k += 1
    
    shape_last = shape_range[np.argmin(gcv)]
    res_last = res[np.argmin(gcv)]
    return(shape_last,gcv,res_last)
