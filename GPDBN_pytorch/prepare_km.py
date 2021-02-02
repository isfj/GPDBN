import numpy as np
import xlwt


def prepare_km(path,name_label,name_time,name_state,save_path,save_name):
    os_time = np.load(path+name_time)
    predict = np.load(path+name_label)
    state = np.load(path+name_state)
    
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('sheet1')
    for i in range(len(os_time)):
        sheet.write(i,0,os_time[i])
    for i in range(len(os_time)):
        if predict[i]<1:
            sheet.write(i,1,0)
        else:
            
            sheet.write(i,1,1)
    for i in range(len(os_time)):
        sheet.write(i,2,state[i])
    workbook.save(save_path+save_name) 





###KM
GPDBN_KM_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/GPDBN_pytorch/km/'
pathomic_km_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/GPDBN_pytorch/pathomic_km/'
name_pre_sur = 'pre_score.npy'
name_ori_time = 'ori_sur.npy'
name_censored = 'censored.npy'
save_name = ['GPDBN_km.xls','pathomic_km.xls']

# prepare_km(GPDBN_KM_path,name_pre_sur,name_ori_time,name_censored,GPDBN_KM_path,save_name[0])
prepare_km(pathomic_km_path,name_pre_sur,name_ori_time,name_censored,pathomic_km_path,save_name[1])