#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:49:56 2024

@author: rstulug
"""

import numpy as np
import os
import h5py
import datetime as dt
from mintpy.utils import readfile
import geopandas



    
def epsg_converter(x,y,old_epsg,new_epsg):
    
    points = geopandas.points_from_xy(x=x,y=y,crs=f'EPSG:{old_epsg}')
    points = points.to_crs(f'EPSG:{new_epsg}')
 
    return (points.x,points.y)
    

def read_def_data(path,date,utm=False,epsg=None,data_type='vel'):
    """
    Parameters
    ----------
    path : str
    path of mintpy folder, eg: /mintpy/asc
    date : int
    chosen date for cumulative deformation eg: 20211220
    utm : boolean, optional
    convert wgs84 to umt coordinates. The default is False.
    epsg : TYPE, optional
    epsg code for projection. The default is None.
    
    Returns
    -------
    final_data : list
    final data lat, lon,height, azimuth,incidence,los, utm y , utm x
    """

    
    fdir = os.path.expanduser(path)
    
    #read time-series and dates
    if data_type =='vel':
        fname = os.path.join(fdir, 'velocity.h5')
        with h5py.File(fname,'r') as f:
            timeseries = f['velocity'][:]
    else:
        fname = os.path.join(fdir, 'timeseries_ERA5_demErr.h5')
        with h5py.File(fname,'r') as f:
            timeseries = f['timeseries'][:]
            dates = f['date'][:]
    
    #chose given date 
    if data_type == 'vel':
        deformation = timeseries
    else:
            
        index  = np.where(dates == np.bytes_(date))[0][0]   
        deformation = timeseries[index][:]
    
    #read mask data
    mask_name = os.path.join(fdir, 'maskTempCoh.h5')
    with h5py.File(mask_name,'r') as f:
        mask = f['mask'][:]
    
    #read azimuth and incidence angle
    geom_file = os.path.join(fdir, 'inputs/geometryGeo.h5')
    with h5py.File(geom_file,'r') as f:
        azimuth = f['azimuthAngle'][:]
        incidence = f['incidenceAngle'][:]
        height = f['height'][:]
        
    #create lat lon grid
    meta = readfile.read_attribute(geom_file)
    x_first = np.float32(meta['X_FIRST'])
    y_first = np.float32(meta['Y_FIRST'])
    x_step = np.float32(meta['X_STEP'])
    y_step = np.float32(meta['Y_STEP'])
    width = np.float32(meta['WIDTH'])
    length =np.float32( meta['LENGTH'])
    
    lats = np.arange(y_first,length*y_step+y_first-y_step,y_step)
    lons = np.arange(x_first,width*x_step+x_first-x_step,x_step)

    lon_mesh,lat_mesh = np.meshgrid(lons,lats)
    
    all_ind = (np.where(mask.flatten()==True)[0])
    
    if utm == True:    
        final_data = np.zeros((len(all_ind),8))
    else:
        final_data = np.zeros((len(all_ind),6))
        
    final_data[:,0] = lat_mesh.flatten()[all_ind]
    final_data[:,1] = lon_mesh.flatten()[all_ind]
    final_data[:,2] = height.flatten()[all_ind]
    final_data[:,3] = azimuth.flatten()[all_ind]
    final_data[:,4] = incidence.flatten()[all_ind]
    final_data[:,5] = deformation.flatten()[all_ind]*1e2
    
    # Add UTM coordinates to the final results
    if utm == True:
        new_x,new_y = epsg_converter(final_data[:,1],final_data[:,0], 4326, epsg)
        # points = geopandas.points_from_xy(x=final_data[:,1],y=final_data[:,0],crs='EPSG:4326')
        # points = points.to_crs(f'EPSG:{epsg}')
        final_data[:,6] = new_y*1e-3
        final_data[:,7] = new_x*1e-3
    return final_data
    


def read_gnss_data(path):
    data = np.loadtxt(path)
    data2 = np.zeros((np.shape(data)[0]*3,np.shape(data)[1]))
    index = 0
    for i in range(0,np.shape(data2)[0],3):
        data2[i,:] = data[index,:]
        data2[i+1,:] = data[index,:]
        data2[i+2,:] = data[index,:]
        index += 1
    return data2


    
def get_unit_vector(azimuth,incidence,d_type='insar'):
    if d_type == 'insar':
        unit_vec = np.array([np.sin(np.deg2rad(azimuth))*np.sin(np.deg2rad(incidence))*-1,np.cos(np.deg2rad(azimuth))*np.sin(np.deg2rad(incidence)),
                    np.cos(np.deg2rad(incidence))])
    else:
        unit_vec = np.array([1,1,1])
    return unit_vec


def design_coeff(def_data,d_type):
    designed_coeff = [None]*len(def_data)
    designed_obs = [None]*len(def_data)
    for i in range(len(def_data)):
        if d_type[i] == 'insar':
            designed_coeff[i] = np.zeros((len(def_data[i]),3))
            # designed_coeff[i] = [-1*np.sin(np.deg2rad(def_data[i][:,2]))*np.sin(np.deg2rad(def_data[i][:,3])),
            #                      np.cos(np.deg2rad(def_data[i][:,2]))*np.sin(np.deg2rad(def_data[i][:,3])),
            #                      np.cos(np.deg2rad(def_data[i][:,3])),def_data[i][:,4]]
            designed_coeff[i][:,0] = -1*np.sin(np.deg2rad(def_data[i][:,3]))*np.sin(np.deg2rad(def_data[i][:,4]))
            designed_coeff[i][:,1] = np.cos(np.deg2rad(def_data[i][:,3]))*np.sin(np.deg2rad(def_data[i][:,4]))
            designed_coeff[i][:,2] = np.cos(np.deg2rad(def_data[i][:,4]))
            designed_obs[i] = def_data[i][:,5]
        else:
            # n_e,n_n,n_u,obs = [],[],[],[]
            designed_coeff[i] = np.zeros((len(def_data[i]),3))
            designed_obs[i] = np.zeros(len(def_data[i]))
            index = 0
            for k in range(0,len(def_data[i]),3):
                designed_coeff[i][k,:] = np.array([1,0,0])
                designed_coeff[i][k+1,:] = np.array([0,1,0])
                designed_coeff[i][k+2,:] = np.array([0,0,1])
                designed_obs[i][k] = def_data[i][k,2]
                designed_obs[i][k+1] = def_data[i][k,3]
                designed_obs[i][k+2] = def_data[i][k,4]
                index += 1
            # designed_obs[i] = np.zeros((len(def_data[i])))
            # for k in range(len(def_data[i])):
            #     for j in range(2,5):
            #         obs.append(def_data[i][k][j])
            #         if j ==2:
            #             n_e.append(1)
            #             n_n.append(0)
            #             n_u.append(0)  
            #         elif j ==3:
            #             n_e.append(0)
            #             n_n.append(1)
            #             n_u.append(0)
            #         else:
            #             n_e.append(0)
            #             n_n.append(0)
            #             n_u.append(1)
            # designed_coeff[i][:,0] = n_e
            # designed_coeff[i][:,1] = n_n
            # designed_coeff[i][:,2] = n_u
            # designed_obs[i] = obs
    return designed_coeff, designed_obs
    
   
def design_control(def_data):
    # n_e,n_n,n_u,obs = [],[],[],[]
    designed_coeff = np.zeros((len(def_data),3))
    designed_obs = []
    for k in range(0,len(def_data),3):
        designed_coeff[k,:] = np.array([1,0,0])
        designed_coeff[k+1,:] = np.array([0,1,0])
        designed_coeff[k+2,:] = np.array([0,0,1])
        designed_obs.append(def_data[k,2])
        designed_obs.append(def_data[k,3])
        designed_obs.append(def_data[k,4])
    # designed_coeff[:,0] = n_e
    # designed_coeff[:,1] = n_n
    # designed_coeff[:,2] = n_u
    # designed_obs = obs

    return designed_coeff,designed_obs
    
    
def coeff_decomp(def_data):
    # n_e,n_n,n_u,obs = [],[],[],[]
    designed_coeff = np.zeros((len(def_data),3))
    for k in range(0,len(def_data),3):
        designed_coeff[k,:] = np.array([1,0,0])
        designed_coeff[k+1,:] = np.array([0,1,0])
        designed_coeff[k+2,:] = np.array([0,0,1])
    return designed_coeff
    
    