from utils import read_pickle, read_points, lhw2wlh
import os

data_root = '/home/abhishek/Desktop/Work/TreeDetection/TreeData_bin'

data_infos = read_pickle(os.path.join(data_root, f'treedata_infos_test.pkl'))
data_infos_filtered = {}

canopy_type = 'high_density'

ids = sorted(data_infos.keys())

for c_name in ids:
    c_name_list = c_name.split('_')
    sub_id = int(c_name_list[1])
    plot_id = int(c_name_list[3])
    
    if ((plot_id%40==39) or (plot_id%40==38) or (plot_id%40==37) or (plot_id%40==36)) and canopy_type=='high_density':
        data_infos_filtered[c_name] = data_infos[c_name]
    
    if ((plot_id%40==0) or (plot_id%40==1) or (plot_id%40==2) or (plot_id%40==3)) and canopy_type=='low_density':
        data_infos_filtered[c_name] = data_infos[c_name]
    
    if ((sub_id==0) or (sub_id==1) or (sub_id==48) or (sub_id==49)) and canopy_type=='specie_specific':
        data_infos_filtered[c_name] = data_infos[c_name]
    
    if ((sub_id==23) or (sub_id==24) or (sub_id==25) or (sub_id==26)) and canopy_type=='specie_mix':
        data_infos_filtered[c_name] = data_infos[c_name]
        

print(data_infos_filtered.keys())
# print(len(data_infos), len(data_infos_filtered))
        
        
    
        
        
    