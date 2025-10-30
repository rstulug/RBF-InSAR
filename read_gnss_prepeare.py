#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:46:53 2024

@author: rstulug
"""

from mintpy.objects import gnss
import numpy as np
import datetime as dt
from utils import read_def_data,epsg_converter
import geopandas
#from mintpy.objects.gnss import GNSS

startDate = '20200101'
endDate='20220101'
type = 'asc'

available_gnss,lat,lon = gnss.search_gnss((40.29,41.20,-112.62,-111.4),start_date=startDate,end_date=endDate)


ref_sta = 'UTDR'
gnss_obj_ref = gnss.get_gnss_class(source='UNR')(ref_sta)
dates_ref,dis_e_ref,dis_n_ref,dis_u_ref,std_e_ref,std_n_ref,std_u_ref= gnss_obj_ref.read_displacement(start_date=startDate,end_date=endDate)

def_data = read_def_data('/mnt/d/Test_Folder/RBF_Insar/saltlake/inputs/dsc',endDate,utm=True,epsg=32612)
################################
# deneme ='AZU1'
# deneme_obj = gnss.get_gnss_class(source='UNR')(deneme)
# deneme_obj.get_los_velocity('/mnt/d/Test_Folder/RBF_Insar/inputs/asc/inputs/geometryGeo.h5',startDate,endDate,gnss_comp='enu2los',ref_site='SPMS')
# #################################
# if type == 'asc':
#     azimuth = np.deg2rad(-12.94876)
#     incidence = np.deg2rad(33.96695)
# else:
#     azimuth = np.deg2rad(-167.07044)
#     incidence = np.deg2rad(39.283763)
#################################

test_gnss = []
control_gnss = []

control_points = ['UTH1','UTM1','UTSL','UTFM','UTGR','UTL2','P116','ALUT','NAIU','UTHE','P478']
test_points = ['UTCR','RBUT','COON','UTWB','P088','UTWV','UTSO','ZLC1','UTBO','UTAI','P114','P115','UTTO','P117','UTEM','UTOC','UTPC','UTHB','UTWA']
remove_points = []

for k,site in enumerate(available_gnss):
    gnss_obj = gnss.get_gnss_class(source='UNR')(site)
    gnss_obj.read_displacement(start_date=startDate,end_date=endDate)
    dates = gnss_obj.dates
    if len(dates) <= (dt.datetime.fromisoformat(endDate)-dt.datetime.fromisoformat(startDate)).days / 50:# or dates[0]>dt.datetime.fromisoformat(startDate) or dates[-1]<dt.datetime.fromisoformat(endDate):
        continue
    else:
        date_last,dis_e,dis_n,dis_u = [],[],[],[]
        date_list  = []
        lat_gnss,lon_gnss = gnss_obj.get_site_lat_lon()
        indexes = np.argmin(np.sqrt((lat_gnss - def_data[:,0])**2 + (lon_gnss - def_data[:,1])**2))
        if gnss_obj.site in control_points:
            for i,date in enumerate(dates_ref):
                if date in dates:
                    index = np.where(dates==date)[0][0]
                    date_last.append(date)
                    date_list.append(date.year+(date.month-1)/12+(date.day-1)/365)
                    dis_e.append(gnss_obj.dis_e[index]-dis_e_ref[i])
                    dis_n.append(gnss_obj.dis_n[index]-dis_n_ref[i])
                    dis_u.append(gnss_obj.dis_u[index]-dis_u_ref[i])
                    # valid_gnss.append([site,lat[i],lon[i],dist_e,dist_n,dist_u,np.sum(dist_e-dist_e[0]),np.sum(dist_n-dist_n[0]),np.sum(dist_u-dist_u[0])])
            date_diff = np.array(date_list)-date_list[0]
            design_mat = np.ones((len(date_diff),2))
            design_mat[:,1] = date_diff
            estimated_e = np.dot(np.linalg.pinv(design_mat),np.array(dis_e-dis_e[0]))[1]
            estimated_n = np.dot(np.linalg.pinv(design_mat),np.array(dis_n-dis_n[0]))[1]
            estimated_u = np.dot(np.linalg.pinv(design_mat),np.array(dis_u-dis_u[0]))[1]
            #control_gnss.append([site,lat[k],lon[k],dis_e,dis_n,dis_u,dis_e[-1]-dis_e[0],dis_n[-1]-dis_n[0],dis_u[-1]-dis_u[0],date_last,indexes])
            control_gnss.append([site,lat[k],lon[k],dis_e,dis_n,dis_u,estimated_e,estimated_n,estimated_u,date_last,indexes])
        else:
            if gnss_obj.site in test_points:
                for i,date in enumerate(dates_ref):
                    if date in dates:
                        index = np.where(dates==date)[0][0]
                        date_last.append(date)
                        date_list.append(date.year+(date.month-1)/12+(date.day-1)/365)
                        
                        dis_e.append(gnss_obj.dis_e[index]-dis_e_ref[i])
                        dis_n.append(gnss_obj.dis_n[index]-dis_n_ref[i])
                        dis_u.append(gnss_obj.dis_u[index]-dis_u_ref[i])
                        # valid_gnss.append([site,lat[i],lon[i],dist_e,dist_n,dist_u,np.sum(dist_e-dist_e[0]),np.sum(dist_n-dist_n[0]),np.sum(dist_u-dist_u[0])])
                
                date_diff = np.array(date_list)-date_list[0]
                design_mat = np.ones((len(date_diff),2))
                design_mat[:,1] = date_diff
                estimated_e = np.dot(np.linalg.pinv(design_mat),np.array(dis_e-dis_e[0]))[1]
                estimated_n = np.dot(np.linalg.pinv(design_mat),np.array(dis_n-dis_n[0]))[1]
                estimated_u = np.dot(np.linalg.pinv(design_mat),np.array(dis_u-dis_u[0]))[1]
                #test_gnss.append([site,lat[k],lon[k],dis_e,dis_n,dis_u,dis_e[-1]-dis_e[0],dis_n[-1]-dis_n[0],dis_u[-1]-dis_u[0],date_last,indexes])
                test_gnss.append([site,lat[k],lon[k],dis_e,dis_n,dis_u,estimated_e,estimated_n,estimated_u,date_last,indexes])
                

test_dis = np.zeros((len(test_gnss),8)) 
for i,data in enumerate(test_gnss): 
    test_dis[i,0] = data[1] 
    test_dis[i,1] = data[2] 
    test_dis[i,2] = data[6]*1e2
    test_dis[i,3] = data[7]*1e2
    test_dis[i,4] = data[8]*1e2
    test_dis[i,5] = (data[6]*np.sin(np.deg2rad(def_data[data[10],3]))*np.sin(np.deg2rad(def_data[data[10],4]))*-1 + data[7]*np.cos(np.deg2rad(def_data[data[10],3]))*np.sin(np.deg2rad(def_data[data[10],4]))+data[8]*np.cos(np.deg2rad(def_data[data[10],4])))*1e2

control_dis = np.zeros((len(control_gnss),8)) 
for i,data in enumerate(control_gnss): 
    control_dis[i,0] = data[1] 
    control_dis[i,1] = data[2] 
    control_dis[i,2] = data[6]*1e2 
    control_dis[i,3] = data[7]*1e2 
    control_dis[i,4] = data[8]*1e2 
    control_dis[i,5] = (data[6]*np.sin(np.deg2rad(def_data[data[10],3]))*np.sin(np.deg2rad(def_data[data[10],4]))*-1 + data[7]*np.cos(np.deg2rad(def_data[data[10],3]))*np.sin(np.deg2rad(def_data[data[10],4]))+data[8]*np.cos(np.deg2rad(def_data[data[10],4])))*1e2

test_x,test_y = epsg_converter(test_dis[:,1],test_dis[:,0], 4326, 32612)
# points = geopandas.points_from_xy(x=final_data[:,1],y=final_data[:,0],crs='EPSG:4326')
# points = points.to_crs(f'EPSG:{epsg}')
test_dis[:,6] = test_y*1e-3
test_dis[:,7] = test_x*1e-3    

   #############################################3
control_x,control_y = epsg_converter(control_dis[:,1],control_dis[:,0], 4326, 32612)
# points = geopandas.points_from_xy(x=final_data[:,1],y=final_data[:,0],crs='EPSG:4326')
# points = points.to_crs(f'EPSG:{epsg}')
control_dis[:,6] = control_y*1e-3
control_dis[:,7] = control_x*1e-3    
   
#deneme = np.loadtxt(f'{type}_def_{endDate}') 
    
import matplotlib.pyplot as plt    
plt.scatter(def_data[:,1],def_data[:,0],c=def_data[:,5],cmap='jet',vmin=-1,vmax=1,s=10**0.1)
plt.scatter(test_dis[:,1],test_dis[:,0],c=test_dis[:,5],cmap='jet',vmin=-1,vmax=1,marker='^',s=10**2,edgecolor='black',linewidths=0.1)
plt.colorbar()
for i,label in enumerate(test_gnss):
    plt.annotate(label[0],(test_dis[i,1],test_dis[i,0]),xytext=(5,5), textcoords='offset points')

#control points
plt.figure(2)
import matplotlib.pyplot as plt    
plt.scatter(def_data[:,1],def_data[:,0],c=def_data[:,5],cmap='jet',vmin=-1,vmax=1,s=10**0.1)
plt.scatter(control_dis[:,1],control_dis[:,0],c=control_dis[:,5],cmap='jet',vmin=-1,vmax=1,marker='^',s=10**2,edgecolor='black',linewidths=0.1)
plt.colorbar()
for i,label in enumerate(control_gnss):
    plt.annotate(label[0],(control_dis[i,1],control_dis[i,0]),xytext=(5,5), textcoords='offset points')




