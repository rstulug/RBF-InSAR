#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 15:26:54 2021

@author: Rasit ULUG
Middle East Technical University

"""

"""
This function constructs design matrix using parallel processing procedure
"""
import numpy as np
import os
import other_functions


def other_kernel_coeff_mat(x,y,x_rbf,y_rbf,shape,workers,designed_coeff,functions):
    storage_size = int(len(x)/10)
    all_coeff = np.zeros((len(x),len(x_rbf)*3))
    if len(x)>storage_size and len(x)>9:
        indicates = np.append(np.arange(0,len(x),storage_size),len(x))
        for i in range(len(indicates)-1):
            # nw_1 = time.time()
            distances = other_functions.dist(x[indicates[i]:indicates[i+1]],y[indicates[i]:indicates[i+1]],x_rbf,y_rbf,workers)    
            coeff_mat = other_functions.multiprocessing_model_coeff_pointmass(distances,shape,designed_coeff[indicates[i]:indicates[i+1]],workers,functions,other_functions.coeff_mat_pointmass)
        #     if i == 0:
        #         f = tables.open_file('coeff_mat.hd5', 'w')
        #         atom = tables.Atom.from_dtype(coeff_mat.dtype)
        #         array_c = f.create_earray(f.root, 'data', atom, (0,len(x_rbf)*3))
                
        #         for idx in range(indicates[i+1]):
        #             array_c.append(coeff_mat)
        #         f.close()
        #     else:
        #         f = tables.open_file('coeff_mat.hd5', mode='a')
        #         f.root.data.append(coeff_mat)
        #         f.close()
        # f = tables.open_file('coeff_mat.hd5', mode='r')
        # coeff_mat = f.root.data[:]
        # f.close()
        # os.remove('coeff_mat.hd5')
            all_coeff[indicates[i]:indicates[i+1],:] = coeff_mat
    else:
        distances = other_functions.dist(x,y,x_rbf,y_rbf,workers) 
        all_coeff = other_functions.multiprocessing_model_coeff_pointmass(distances,shape,designed_coeff,workers,functions,other_functions.coeff_mat_pointmass)
    return(all_coeff)
