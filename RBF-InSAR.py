#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 15:26:54 2021

@author: Rasit_ULUG
Middle East Technical University
"""
# Importing modules
import os
import sys
import time
import multiprocessing
import numpy as np
import warnings
import gc
import coeff_matrix
import kernel_functions
import other_functions
import k_SRBF
import utils
from utils import read_def_data, epsg_converter,read_gnss_data,design_coeff,design_control

warnings.filterwarnings('ignore')
nw_first = time.time()
##############################################################################
print("\nModeling 3D InSAR data by using Radial Basis Functions \n")
##############################################################################
utm_converter = 1
epsg = 32610
total_data = 0

data_number = 1#int(input('\nPlease enter how many different type of data you will use for modeling: '))
utm_converter = 1#int(input('Enter 1 if you want convert wgs84 to utm: '))
if utm_converter == 1:
    epsg = 32610#int(input('Enter epsg code to convert wgs84 to utm: '))

#Empty list definition to store data
def_data = [None]*data_number
d_type = [None]*data_number
#datas = ['inputs/asc','inputs/dsc','inputs/gnss_test.txt']
#data_types= [1,1,2]

datas = ['inputs/dsc']
data_types= [1]
#Importing data
for i in range(data_number):
    data_type =data_types[i]# dataint(input('\n Please enter data type 1 for Insar 2 for GNSS: '))
    if data_type == 1:
        d_type[i] ='insar'
    
        print('\nPlease import {}. data file: '.format(i+1))
        print('\nThe folder must contain mintpy output files')
        data_path = datas[i]#str(input('{}. Data File: '.format(i+1)))
        date = '20211220'#str(input('Please import deformation date: '))
    else:
        d_type[i] ='gnss'
        print('\nPlease import {}. data file: '.format(i+1))
        print('\nThe file contains lat, lon, dis_e cm , dis_n cm, dis_u cm , los m')
        #data_path = 'inputs/gnss_test_asc.txt'#str(input('{}. Data File: '.format(i+1)))
    if data_type == 1:
        try:
            def_data[i] = read_def_data(data_path,date,utm_converter==1,epsg)
            total_data += len(def_data[i])
        except:
            raise AttributeError("File does not exist or file format is wrong!")
    else:
        try:
            def_data[i] = read_gnss_data(data_path)
            total_data += len(def_data[i])
        except:
            raise AttributeError("File does not exist or file format is wrong!")
        

##############################################################################
control_data = 2
print('\nDo you want to use external control points to validate results')
print('\nNote: If you want to use turning point algorithm, you have to give external control points')
control_data = 1#int(input('\n1: Yes \n2: No  \n: '))

if control_data not in (1,2):
    raise ValueError('You choose wrong control data option. Enter 1 to use external control data, 2 for No')

if control_data == 1:
    print('\nPlease import control data file: ')
   
    control_file = 'inputs/gnss_control_asc.txt'#input('\nControl Points File: ')
    try:
        control_data = read_gnss_data(control_file)
    except:
        raise AttributeError("File does not exist or file format is wrong!")  
    

    control_data_type = 'gnss'
#####################################################
# for i in range(data_number):
#     def_data[i][:,7] *= 1e-2
#     def_data[i][:,6] *= 1e-2

# control_data[:,7] *= 1e-2
# control_data[:,6] *= 1e-2
##############################################################################
designed_coeff,designed_obs = design_coeff(def_data,d_type)
designed_control,designed_control_obs = design_control(control_data)

##############################################################################
if np.any(np.array(data_types)==2):
    data_number += 1
    for i,k in enumerate(data_types):
        if k == 2:
            index_for_gnss_u = np.arange(2,len(def_data[i]),3)
            def_data.append(def_data[i][index_for_gnss_u,:])
            def_data[i] = np.delete(def_data[i],index_for_gnss_u,axis=0)
            
            designed_coeff.append(designed_coeff[i][index_for_gnss_u,:])
            designed_coeff[i] = np.delete(designed_coeff[i],index_for_gnss_u,0)
            
            designed_obs.append(designed_obs[i][index_for_gnss_u])
            designed_obs[i] = np.delete(designed_obs[i],index_for_gnss_u,0)

##############################################################################
print('\nDefine parameters for k-SRBF algorihtm ')

k_init = 1#int(input('\nPlease enter the k number for the construction of the initial candidate network: '))
min_p  = 4#int(input('\nPlease enter the minimum sample number for each cluster: ')) 
min_dist = 1.2#float(input('\nPlease enter the minimum euclidian distance between the centroids (km)): '))
#max_srbf =0 float(input('\nPlease enter the maximum spherical distance between centroid and its samples to split the cluster (degree)): '))
##############################################################################

workers = 56
    
gc.collect() 
##############################################################################
print('\n\nHere we go...')
##############################################################################
#Functional model selection depending on the choosen kernel type
coeff_mat = [None]*data_number

function = kernel_functions.gaussian_kernel

# function = kernel_functions.thin_plate_spline

# ##############################################################################
print('\nk-SRBF algorithm is initialized. It can take some time. Please wait...')
nw = time.time()
x_rbf,y_rbf = k_SRBF.k_SRBF(def_data,k_init,min_p,min_dist,workers)
gc.collect()  
print('\nThe data-adaptive network design completed. Process time:{:.2f} minute'.format((time.time()-nw)/60))     

lon_rbf,lat_rbf = epsg_converter(x_rbf*1e3,y_rbf*1e3,32610,4326)     

np.savez('outputs/rbf_1.2',x_rbf=x_rbf,y_rbf=y_rbf,lon_rbf=lon_rbf,lat_rbf=lat_rbf)
##############################################################################
# rbfs = np.load('inputs/rbf_1.2.npz')
# x_rbf = rbfs['x_rbf']
# y_rbf = rbfs['y_rbf']
# lon_rbf = rbfs['lon_rbf']
# lat_rbf = rbfs['lat_rbf']
##############################################################################  
if function == kernel_functions.gaussian_kernel:
    ria = 3
    shape_i = 0.1
    shape_l =1
    shape_s = 0.1
    shape_range = np.arange(shape_i,shape_l+shape_s,shape_s)
#     ##############################################################################
    regu = 0
    shapes = np.zeros(len(x_rbf))
    
    if d_type.count('insar') == 2:
    
        all_x = [np.concatenate([def_data[0][:,7],def_data[1][:,7]])]
        all_y = [np.concatenate([def_data[0][:,6],def_data[1][:,6]])]
        all_designed_coeff = [np.concatenate([designed_coeff[0],designed_coeff[1]])]
        
        all_res2 = [np.concatenate([designed_obs[0],designed_obs[1]])]
    elif d_type.count('insar') == 1:
        all_x = [np.concatenate([def_data[0][:,7]])]
        all_y = [np.concatenate([def_data[0][:,6]])]
        all_designed_coeff = [np.concatenate([designed_coeff[0]])]
        
        all_res2 = [np.concatenate([designed_obs[0]])]
    else:
        raise ValueError('Olmadı reis')
    
        
    for i in range(len(x_rbf)):
    
        
        index1 = np.intersect1d(np.where(all_x[0]<=x_rbf[i]+ria)[0],np.where(all_x[0]>=x_rbf[i]-ria)[0])
        index2 = np.intersect1d(np.where(all_y[0]<=y_rbf[i]+ria)[0],np.where(all_y[0]>=y_rbf[i]-ria)[0])
        index = np.intersect1d(index1,index2)
        
        x_2 = [all_x[0][index]]
        y_2 = [all_y[0][index]]
        designed_coeff_2 = [all_designed_coeff[0][index,:]]
        designed_obs_2 = [all_res2[0][index]]        
            
        total_data = len(index)    
    
        shape_last,gcv,deformation = k_SRBF.shape_sel_with_GCV_ind(x_2,y_2,np.array([x_rbf[i]]),np.array([y_rbf[i]]),designed_coeff_2,designed_obs_2,workers,shape_range,total_data,function)
        print('Depth selected as {:.1f} for {:1d} SRBF'.format(shape_last,i))
        shapes[i] = shape_last
        
        all_res2[0][index] -= deformation
    #############################################################################
    np.save('outputs/shape_0.1_1_0.1_0.5',shapes)
else:
    shapes = np.ones(len(x_rbf))

##############################################################################

#shape_last,gcv = k_SRBF.shape_sel_with_GCV(def_data,x_rbf,y_rbf,designed_coeff,designed_obs,workers,shape_range,total_data,function)
##############################################################################      
#shapes = np.ones(len(x_rbf))*shape_last
#np.save('inputs/shape_0.1_1_0.1_0.6',shapes)
##############################################################################    

# print('\nFinal Coefficient Matrix Calculation. Please wait...')
# all_dist = [None]*data_number
# for i in range(data_number):
#     all_dist[i] = other_functions.dist(def_data[i][:,7], def_data[i][:,6], x_rbf, y_rbf, workers)    

for i in range(data_number):
    coeff_mat[i] = coeff_matrix.other_kernel_coeff_mat(def_data[i][:,7],def_data[i][:,6],x_rbf,y_rbf,shapes,workers,designed_coeff[i],function)

#cond = np.linalg.cond(np.dot(coeff_mat[0].T,coeff_mat[0]))
#print(cond)
#cond < 1/sys.float_info.epsilon
regu = 1
print('Coefficient matrix calculation done. Process time: {:.2f} minute'.format((time.time()-nw)/60))
print('\nParameter Estimation started. Please wait...')   
nw = time.time()
if regu == 0:
    parameter = other_functions.LS(coeff_mat,designed_obs)
else: 
    parameter,sigma = other_functions.MCVCE(coeff_mat,designed_obs) 

np.save('outputs/parameter',parameter)
coeff_mat = [None]*data_number
gc.collect()
print('\nParameter Estimation completed.Process time: {:.2f} minute'.format((time.time()-nw)/60))
##############################################################################
# External Validation (Control Points)

print('\nExternal validation for gravity control points started...')
nw = time.time()

distance_test = other_functions.dist(control_data[:,7],control_data[:,6],x_rbf,y_rbf,workers)


    
coeff_mat_test = other_functions.multiprocessing_model_coeff_pointmass(distance_test,shapes,designed_control,workers,function,other_functions.coeff_mat_pointmass)
print('Test coefficient matrix calculation done.Process time:{:.2f} minute'.format((time.time()-nw)/60))

estimated_test = np.dot(coeff_mat_test,parameter)
residual_test = designed_control_obs - estimated_test

res_e_c,res_n_c,res_u_c =np.zeros(int(len(residual_test)/3)),np.zeros(int(len(residual_test)/3)),np.zeros(int(len(residual_test)/3))
index = 0
for i in range(0,len(residual_test),3):
    res_e_c[index] = residual_test[i]
    res_n_c[index] = residual_test[i+1]
    res_u_c[index] = residual_test[i+2]
    index += 1
    

rms_e_c = np.sqrt(np.sum(res_e_c**2)/len(res_e_c))
rms_n_c = np.sqrt(np.sum(res_n_c**2)/len(res_n_c))
rms_u_c = np.sqrt(np.sum(res_u_c**2)/len(res_u_c))
coeff_mat_test = None
print('\nRMS of East: {:.3f} cm'.format(rms_e_c))
print('max {:.3f}'.format(np.max(res_e_c)))
print('min {:.3f}'.format(np.min(res_e_c)))
print('std {:.3f}'.format(np.std(res_e_c)))
print('mean {:.3f}'.format(np.mean(res_e_c)))


print('\nRMS of North: {:.3f} cm'.format(rms_n_c))
print('max {:.3f}'.format(np.max(res_n_c)))
print('min {:.3f}'.format(np.min(res_n_c)))
print('std {:.3f}'.format(np.std(res_n_c)))
print('mean {:.3f}'.format(np.mean(res_n_c)))


print('\nRMS of Up: {:.3f} cm'.format(rms_u_c))
print('max {:.3f}'.format(np.max(res_u_c)))
print('min {:.3f}'.format(np.min(res_u_c)))
print('std {:.3f}'.format(np.std(res_u_c)))
print('mean {:.3f}'.format(np.mean(res_u_c)))

parameter_e, parameter_n,parameter_u = np.zeros(int(len(parameter)/3)),np.zeros(int(len(parameter)/3)),np.zeros(int(len(parameter)/3))


indis = 0
for i in range(0,len(parameter),3):
    parameter_e[indis] = parameter[i]
    parameter_n[indis] = parameter[i+1]
    parameter_u[indis] = parameter[i+2]
    indis += 1
 
distance_decomposed = other_functions.dist(def_data[0][:,7],def_data[0][:,6],x_rbf,y_rbf,workers)
designed_decomposed = np.ones(len(def_data[0]))
coeff_decomposed = other_functions.multiprocessing_model_coeff_pointmass(distance_decomposed,shapes,designed_decomposed,workers,function,other_functions.coeff_mat_pointmass)

east = np.dot(coeff_decomposed,parameter_e)
north = np.dot(coeff_decomposed,parameter_n)
up = np.dot(coeff_decomposed,parameter_u)
# designed_coeff_decomposed = coeff_decomp(def_data[1]) 

# designed_coeff_decomposed2 = np.vstack((designed_coeff_decomposed ,np.vstack((designed_coeff_decomposed ,designed_coeff_decomposed ))))


# designed_x = np.zeros(len(def_data[1][:,7])*3)
# designed_y = np.zeros(len(def_data[1][:,6])*3)

# indis = 0
# for i in range(0,len(designed_x),3):
#     designed_x[i] = def_data[1][indis,7]
#     designed_x[i+1] = def_data[1][indis,7]
#     designed_x[i+2] = def_data[1][indis,7]

#     designed_y[i] = def_data[1][indis,6]
#     designed_y[i+1] = def_data[1][indis,6]
#     designed_y[i+2] = def_data[1][indis,6]
#     indis +=1
    
    
# distance_decomposed = other_functions.dist(designed_x,designed_y,x_rbf,y_rbf,workers)
# coeff_mat_decomposed = other_functions.multiprocessing_model_coeff_pointmass(distance_decomposed,shapes,designed_coeff_decomposed2,workers,function,other_functions.coeff_mat_pointmass)



# decomposed = np.dot(coeff_mat_decomposed,parameter)

# coeff_mat_decomposed = None
# gc.collect()

# east,north,up =np.zeros(int(len(def_data[1]))),np.zeros(int(len(def_data[1]))),np.zeros(int(len(def_data[1])))
# index = 0
# for i in range(0,len(decomposed),3):
#     east[index] = decomposed[i]
#     north[index] = decomposed[i+1]
#     up[index] = decomposed[i+2]
#     index += 1
    
np.save('outputs/east_comp',east)
np.save('outputs/north_comp',north)
np.save('outputs/up_comp',up)