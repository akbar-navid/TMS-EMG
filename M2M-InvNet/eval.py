# %% Import libraries and set seeds
from utils import *
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import io
from scipy import stats
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
#from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt

seed_value= 0
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)

tf.keras.backend.clear_session()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
%matplotlib inline
tf.compat.v1.disable_eager_execution()
init_op = tf.compat.v1.global_variables_initializer()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.compat.v1.Session(config=config)
run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
sess.run(init_op)
   

# %% Argparse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='/home/navid/SDrive/CSL/2022/DT_TMS_EMG')
parser.add_argument('--sub', type=int, default=3)
args = parser.parse_args()

# %% [Data load: Sub-1] below
if args.sub==1:

    n_files          = 699
    # ds = 1
    img_shape        = (64, 64, 64)
    MEP_shape        = (15)
    data_type        = np.float32
    X = np.zeros((n_files, *img_shape, 1), dtype = data_type) 
    Y = np.zeros((n_files, 15, 1), dtype = data_type) 
    print('Y:',Y.shape)

    # Load E-field Stims (Sub-MD, All RMTs)
    m = 0
    thresh = 4e-4
    for i in range(n_files):     
        tmp_mat = io.loadmat('data_MD//Masked_All_RMT//{0}.mat'.format(i+1)) 
        X[m,:,:,:,0] = np.array(tmp_mat['zone'], dtype=data_type)     
        m+=1   

    # Load MEPs (Sub-MD, All RMTs)
    tmp_mat = io.loadmat('data_MD//MD_RMT_110_120_130_140.mat') 
    Y = np.array(tmp_mat['MD_RMT_110_120_130_140'], dtype=data_type)  
        
    # Load Only Normal Stims
    normal_stims = []
    for i in range(n_files):      
        a = np.amax(X[i])
        if a < thresh:       
            continue
        else:
            normal_stims.append(i)
        
    print(len(normal_stims))
    Y_norm = Y[normal_stims]
    X_norm= X[normal_stims]

    # Discard Zero MEP Stims
    Y_ind = np.any(Y_norm >= 1e-3, axis=1) 
    Y_train = Y_norm[Y_ind]
    X_train = X_norm[Y_ind]
            
    # Min-Max Scaling
    a = np.amax(X_train)
    b = np.amin(X_train)
    X_train = (X_train-b)/(a-b)

    print(a,b)
    print(np.amax(X_train),np.amin(X_train))

    print ("X shape: " + str(X_train.shape))
    print ("Y shape: " + str(Y_train.shape))

    # Load Mask
    tmp_mat = io.loadmat('data_MD//M1_mask.mat')
    motor_mask = tmp_mat['mask']
    motor_mask = motor_mask.reshape(1,*img_shape,1)
    motor_mask = tf.convert_to_tensor(motor_mask, np.float32)

# %% [Data load: Sub-2] below
if args.sub==2:

    n_files          = 1196
    # ds = 1
    img_shape        = (64, 64, 64)
    MEP_shape        = (14,)
    # MEP_shape        = (15,)
    data_type        = np.float32
    X = np.zeros((n_files, *img_shape, 1), dtype = data_type) 
    Y = np.zeros((n_files, *MEP_shape, 1), dtype = data_type) 

    # Load E-field Stims (Sub-025, All RMTs)
    k = [55,    61,   150,   254]
    m = 0

    # Load E-field Stims (Sub-025, All RMTs)
    for i in range(1200):
        j = i + 1
        if j not in k:    
            if j<1000:
                tmp_mat = io.loadmat('data_025/output/E-box/0{0}.mat'.format(j)) 
                X[m,:,:,:,0] = np.array(tmp_mat['box_matrix_E'], dtype=data_type)     
                m += 1   
            else:
                tmp_mat = io.loadmat('data_025/output/E-box/{0}.mat'.format(j)) 
                X[m,:,:,:,0] = np.array(tmp_mat['box_matrix_E'], dtype=data_type)     
                m += 1    

    ## Load MEPs (Sub-025, All RMTs)
    tmp_mat = io.loadmat('data_025/output/MEPampNormAll_MBY.mat') 
    Y = np.array(tmp_mat['MEPampNormAll_MBY'], dtype=data_type)  
    print('Y:',Y.shape)
        
    # Load Only Normal Stims
    normal_stims = []
    for i in range(n_files):      
        a = np.amax(X[i])
        if a < 4e-4:       
            continue
        else:
            normal_stims.append(i)
        
    print(len(normal_stims))
    Y_norm = Y[normal_stims]
    X_norm= X[normal_stims]

    # Discard Zero MEP Stims
    Y_ind = np.any(Y_norm >= 1e-3, axis=1) 
    Y_train = Y_norm[Y_ind]
    X_train = X_norm[Y_ind]
            
    # Min-Max Scaling
    a = np.amax(X_train)
    b = np.amin(X_train)
    X_train = (X_train-b)/(a-b)
            
    # # Mean Standardization
    # a = np.mean(X_train)
    # b = np.std(X_train)
    # X_train = (X_train-a)/b

    print(a,b)
    print(np.amax(X_train),np.amin(X_train))

    print ("X shape: " + str(X_train.shape))
    print ("Y shape: " + str(Y_train.shape))

    # Load Mask
    tmp_mat = io.loadmat('data_025/output/mask_025.mat')
    motor_mask = tmp_mat['logical_mask']
    motor_mask = motor_mask.reshape(1,*img_shape,1)
    motor_mask = tf.convert_to_tensor(motor_mask, np.float32)


# %% [Data load: Sub-3] below
if args.sub==3:

    n_files          = 1199
    # ds = 1
    img_shape        = (64, 64, 64)
    MEP_shape        = (15)
    data_type        = np.float32
    X = np.zeros((n_files, *img_shape, 1), dtype = data_type) 
    Y = np.zeros((n_files, 15, 1), dtype = data_type) 

    # Load E-field Stims (Sub-MD, All RMTs)
    m = 0
    thresh = 4e-4
    # j = [300,]
    # k = [range(1208)]
    list3 = []
    # list3 = [16,26,32,46]
    calc = 113
    q = 1320
    list_int = [300,601,602,603,604,605,606,907,1021]
    list2 = []
    for i in list_int:
        list2.append(i+113)
    # print(list2)

    while calc <= q:
        if calc in list2:
            calc += 1  
            continue
        else:
            list3.append(calc)
            calc += 1           
    # print(len(list3))

    for i in list3:     
        if i<1000:
            tmp_mat = io.loadmat(args.dir+'/data_026/E-box/026_E-box_0{0}.mat'.format(i)) 
            X[m,:,:,:,0] = np.array(tmp_mat['box_matrix_E'], dtype=data_type)     
            m+=1   
        else:
            tmp_mat = io.loadmat(args.dir+'/data_026/E-box/026_E-box_{0}.mat'.format(i)) 
            X[m,:,:,:,0] = np.array(tmp_mat['box_matrix_E'], dtype=data_type)     
            m+=1       

    # Load MEPs (Sub-MD, All RMTs)
    tmp_mat = io.loadmat(f'{args.dir}/data_026/Y_026.mat') 
    Y = np.array(tmp_mat['MEPampNormAll'], dtype=data_type)  
    # print('Y:',Y.shape)
        
    # Load Only Normal Stims
    normal_stims = []
    for i in range(n_files):      
        a = np.amax(X[i])
        if a < thresh:       
            continue
        else:
            normal_stims.append(i)
        
    # print(len(normal_stims))
    Y_train = Y[normal_stims]
    X_train= X[normal_stims]

    # # Discard Zero MEP Stims
    # Y_ind = np.any(Y_norm >= 1e-3, axis=1) 
    # Y_train = Y_train[Y_ind]
    # X_train = X_train[Y_ind]
            
    # Min-Max Scaling
    a = np.amax(X_train)
    b = np.amin(X_train)
    X_train = (X_train-b)/(a-b)

    # print(a,b)
    # print(np.amax(X_train),np.amin(X_train))

    print ("X shape: " + str(X_train.shape))
    print ("Y shape: " + str(Y_train.shape))

    # Load Mask
    tmp_mat = io.loadmat(f'{args.dir}/data_026/mask_026.mat')
    motor_mask = tmp_mat['logical_mask']
    motor_mask = motor_mask.reshape(1,*img_shape,1)
    motor_mask = tf.convert_to_tensor(motor_mask, np.float32)

# %% Model [Direct Variational] Eval below

nrmse_all, r_sq_all, rmt_level = [], [], [] 
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):

    if j>0:
        continue
    else:
        print('\n Running Fold: ', j+1)
        X_train_CV, Y_train_CV = X_train[train_idx], Y_train[train_idx]  
        X_test_CV, Y_test_CV = X_train[test_idx], Y_train[test_idx]  
                   
        mep = Input(shape=MEP_shape)
        map_rev_vae = None
        
        img_map, loss_map = Inv_Map_VAE_Lat_2d(inputs=mep,motor_mask=motor_mask,l_1=1e-4,act_f=relu_out) 
    
##        Sub-1 
#         run_str_map = 'Inv_Map_Lat,enc_map,no_weights,l1,Sep_29,fold_'+str(j+1)
    
##        Sub-2
        # run_str_map = '025,Variational,l1,Oct_01,fold_'+str(j+1)
    
        if args.sub==3: 
            run_str_map = '026,Variational,l1,Jan_2023,fold_'+str(j+1)

        map_rev_vae = Model(mep, img_map)      
        map_rev_vae.load_weights('weights//'+run_str_map+'.h5', by_name=True)
        decoded_imgs_vae_enc = map_rev_vae.predict(Y_test_CV)        
        diff_vae_enc_abs = abs(X_test_CV - decoded_imgs_vae_enc)       
        
        for i in range(Y_test_CV.shape[0]):
            X_nrmse = decoded_imgs_vae_enc[i]   
            nrmse, r_sq, _ = calc_nrmse_r_sq(X_nrmse,X_test_CV[i])
            nrmse_all.append(nrmse), r_sq_all.append(r_sq)
#             rmt_level.append(find_RMT(Y_test_CV[i],Y,299,448,598))   # Subject-1
            rmt_level.append(find_RMT(Y_test_CV[i],Y,295,595,895))    # Subject-2     
#             rmt_level.append(find_RMT(Y_test_CV[i],Y,299,598,898))    # Subject-3           
        
rmt_level, nrmse_all, r_sq_all = np.array(rmt_level),np.array(nrmse_all),np.array(r_sq_all)    
a, b, c, d = [], [], [], []
for i in np.unique(rmt_level):
    print('For RMT-',i)
    ind = np.argwhere(rmt_level==i)
    a.append(np.mean(nrmse_all[ind])), b.append(1.96*stats.sem(nrmse_all[ind],ddof=0))
    c.append(np.mean(r_sq_all[ind])), d.append(1.96*stats.sem(r_sq_all[ind],ddof=0))   
a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)    
print(a,'\n', b,'\n', c,'\n', d)

# %% [markdown]
# # [Temp: Sub-1] below

# %%
# io.savemat('matlab/026/Feb_16/Y_test_f3.mat',                    # ---Normal Y_test---
#            mdict={'Y_test_f3': Y_test_CV}) 

for i in range (X_test_CV.shape[0]):
    io.savemat('matlab/026/Feb_16/X_test_f3'+'_'+str(i+1)+'.mat',                           # ---Normal X_test---#
               mdict={'X_test_f3': X_test_CV[i].reshape(64,64,64)})

#     io.savemat('matlab//026/Feb_16/X_AE_dec'+'_'+str(i+1)+'.mat',                   # ---AE_Dec---#
#                mdict={'X_AE_dec': decoded_imgs_ae[i].reshape(64,64,64)})            
    
#     io.savemat('matlab//026/Feb_16/X_Conv'+'_'+str(i+1)+'.mat',                   # ---Convolutional---#
#                mdict={'X_Conv': decoded_imgs_ae[i].reshape(64,64,64)})      
    
#     io.savemat('matlab/026/Feb_16/X_VAE_dec'+'_'+str(i+1)+'.mat',                           # ---VAE_Dec---#
#                mdict={'X_VAE_dec': decoded_imgs_vae_enc[i].reshape(64,64,64)})        
    
#     io.savemat('matlab//026/Feb_16/X_VAE_mod'+'_'+str(i+1)+'.mat',                   # ---VAE_Mod---#
#                mdict={'X_VAE_mod': decoded_imgs_vae_enc[i].reshape(64,64,64)})          
    
#     io.savemat('matlab//026/Feb_16/X_Var'+'_'+str(i+1)+'.mat',                   # ---Variational---#
#                mdict={'X_Var': decoded_imgs_vae_enc[i].reshape(64,64,64)})          

# %% [markdown]
# # [MATLAB Files Saved] above & [R_sq & RMT eval] below

# %% 

img_shape        = (64, 64, 64)
data_type        = np.float32
X_nrmse = np.zeros((1, *img_shape, 1), dtype = data_type) 
NRMSE_all = []
cos_sim_all = []
R_sq_all = []
# samp_no = 23

# list3 = []
# # # list3 = [16,26,32,46]

# calc = 89
# q = 116

# while calc < q:
#     list3.append(calc)
#     calc += 1    

list3 =  range(len(Y_test_CV))

# for q in range(samp_no):

for q in list3:
    i = q + 1

#     tmp_mat = io.loadmat('matlab/026/Feb_16/X_Var_{0}.mat'.format(i)) 
#     X_nrmse[0,:,:,:,0] = np.array(tmp_mat['X_Var'], dtype=data_type)   

#     tmp_mat = io.loadmat('matlab/026/Feb_16/X_VAE_mod_{0}.mat'.format(i)) 
#     X_nrmse[0,:,:,:,0] = np.array(tmp_mat['X_VAE_mod'], dtype=data_type) 

#     tmp_mat = io.loadmat('matlab/026/Feb_16/X_VAE_dec_{0}.mat'.format(i)) 
#     X_nrmse[0,:,:,:,0] = np.array(tmp_mat['X_VAE_dec'], dtype=data_type) 

#     tmp_mat = io.loadmat('matlab/026/Feb_16/X_Conv_{0}.mat'.format(i)) 
#     X_nrmse[0,:,:,:,0] = np.array(tmp_mat['X_Conv'], dtype=data_type) 

    tmp_mat = io.loadmat('matlab/026/Feb_16/X_AE_dec_{0}.mat'.format(i)) 
    X_nrmse[0,:,:,:,0] = np.array(tmp_mat['X_AE_dec'], dtype=data_type) 

    mse = np.mean((X_test_CV[i-1] - X_nrmse)**2)
    img_pow = np.mean(X_test_CV[i-1]**2)
    NRMSE = np.sqrt(mse/img_pow)
    print('NRMSE - trial %d : %.4f' %(i,NRMSE))  
    NRMSE_all.append(NRMSE)

    a = X_nrmse.reshape(262144,1)
    b = X_test_CV[i-1].reshape(262144,1)

#     x = np.zeros(10968, dtype = data_type) 
#     y = np.zeros(10968, dtype = data_type) 
#     a_ind = 0
#     b_ind = 0

#     for l in range(262144):   
#         if a[l] > 0.:  
#     #         x.append(a[i])
#             x[a_ind] = a[l]
#             a_ind += 1

#     for l in range(262144):   
#         if b[l] > 0.:     
#             y[b_ind] = b[l]
#             b_ind += 1       

#     corr,_ = stats.pearsonr(x,y)
# #     print(corr)
#     R_sq = corr**2
#     R_sq_all.append(R_sq)
#     print('R-sqaured - trial %d : %.4f' %(i,R_sq)) 
#     # print(np.cov(x, y))
#     # np.corrcoef(x,y)
    
print('NRMSE_mean: %.3f' %(np.mean(NRMSE_all)))
print('NRMSE_std: %.3f' %(np.std(NRMSE_all)))
# print('R-squared_mean: %.3f' %(np.mean(R_sq_all)))
# print('R-squared_std: %.3f' %(np.std(R_sq_all)))

# %%
## print(Y_test_CV.shape)
# print(len(Y_test_CV)) 

# print(np.argwhere(np.all((Y-Y_test_CV[22])==0, axis=1))) # Sample for RMT-110 [19, 259] .235
# print(np.argwhere(np.all((Y-Y_test_CV[62])==0, axis=1))) # Sample for RMT-120 [27, 390] .226
# print(np.argwhere(np.all((Y-Y_test_CV[88])==0, axis=1))) # Sample for RMT-130 [39, 528] .166
# print(np.argwhere(np.all((Y-Y_test_CV[46])==0, axis=1))) # Sample for RMT-140 [47, 691] .113
# # print(x)

# # stat = [.218,.2198,.2178,0.22165875, 0.252373, 0.23301996]
# # stat = [.226,.236,.246,0.256]
# # stat = [.231,.241,.251,0.255]
# stat = [0.3570811, 0.36155155, 0.37128767, 0.35990828, 0.369859, 0.36600775]
# # stat = [.1866,.1842,.1875]
# # stat = [0.854,0.814,0.895,0.870]

# # print(stat.**2)
# print(np.mean(stat))
# print(np.std(stat))

# # stat = [0.163,0.186,0.085,0.142]
# # var = 0
# # for m in stat:
# # #     print(m**2)
# #     var += m**2
# # print(np.sqrt(var/4))

# # x = x.astype(float)
# #     y = np.asarray(y)
# # tf.keras.backend.clear_session()

print('NRMSE_mean: ', np.mean(nrmse_all))
print('NRMSE_SEM: ', 1.96*stats.sem(nrmse_all,ddof=0))
print('R-squared_mean: ', np.mean(r_sq_all))
print('R-squared_SEM: ', 1.96*stats.sem(r_sq_all,ddof=0))
# print(r_sq_all)



# %% [markdown]
# # [Metrics Across Folds] below

# %%
#             rmt_level.append(find_RMT(Y_test_CV[i],Y,299,448,598))   # Subject-1
# rmt_level.append(find_RMT(Y_test_CV[i],Y,295,595,895))    # Subject-2     
#             rmt_level.append(find_RMT(Y_test_CV[i],Y,299,598,898))    # Subject-3           

rmt_level, nrmse_all, r_sq_all = np.array(rmt_level),np.array(nrmse_all),np.array(r_sq_all)    
a, b, c, d = [], [], [], []
for i in np.unique(rmt_level):
    print('For RMT-',i)
    ind = np.argwhere(rmt_level==i)
    a.append(np.mean(nrmse_all[ind])), b.append(1.96*stats.sem(nrmse_all[ind],ddof=0))
    c.append(np.mean(r_sq_all[ind])), d.append(1.96*stats.sem(r_sq_all[ind],ddof=0))   
a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)    
print(a,'\n', b,'\n', c,'\n', d)

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

rmse_all = []
nrmse_all = []
MEP_av = []
MEP_active_av = []
MEP_var = []
MEP_active_var = []
Muscles_Active = []
box_data = []
Y_index = []
RMT_level = []
nrmse_lowest = 1.
nrmse_highest = 0.
nrmse_avg = 0.157
eps = 1.
path = "../../../Dropbox/Summer \'20/TMS_SD/TNSRE/May_16/matlab/"

mep = Input(shape=(15,))
img_map, loss_map = Inv_Map_VAE_Lat_2d(inputs=mep,l_1=1e-4,act_f=relu_out) 
map_rev_vae = Model(mep, img_map)   

folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):
    print('\n Running Fold: ', j+1)
    X_train_CV = X_train[train_idx]  
    Y_train_CV = Y_train[train_idx]  
    X_test_CV = X_train[test_idx]  
    Y_test_CV = Y_train[test_idx]          

    run_str_map = '026,Variational,no_weights,l1,Jan_20,fold_'+str(j+1)
    map_rev_vae.load_weights('weights//'+run_str_map+'.h5', by_name=True)
    decoded_imgs_vae_enc = map_rev_vae.predict(Y_test_CV)        
    diff_vae_enc_abs = abs(X_test_CV - decoded_imgs_vae_enc)     

    NRMSE_all = []
    RMSE_all = []
    R_sq_all = []

    for i in range(Y_test_CV.shape[0]):
        X_nrmse = decoded_imgs_vae_enc[i]            
        mse = np.mean((X_test_CV[i] - X_nrmse)**2)
        RMSE_all.append(np.sqrt(mse))
        img_pow = np.mean(X_test_CV[i]**2)
        NRMSE = np.sqrt(mse/img_pow)
        NRMSE_all.append(NRMSE)
        Y_index.append(np.argwhere(np.all((Y_train-Y_test_CV[i])==0, axis=1)))
        RMT_level.append(find_RMT(Y_test_CV[i],Y_norm,299,598,898))
#         if NRMSE<nrmse_lowest:
#             nrmse_lowest = NRMSE
#             io.savemat(path+'X_GT_lowest'+'.mat',mdict={'X_GT_lowest': X_test_CV[i].reshape(64,64,64)})
#             io.savemat(path+'X_recon_lowest'+'.mat',mdict={'X_recon_lowest': X_nrmse.reshape(64,64,64)})
#             io.savemat(path+'diff_lowest'+'.mat',mdict={'diff_lowest': diff_vae_enc_abs[i].reshape(64,64,64)})
#             y_nrmse_lowest = np.argwhere(np.all((Y_norm-Y_test_CV[i])==0, axis=1))
#             r_sq_lowest = calc_R_sq(X_nrmse,X_test_CV[i])
#         if NRMSE>nrmse_highest:
#             nrmse_highest = NRMSE
#             io.savemat(path+'X_GT_highest'+'.mat',mdict={'X_GT_highest': X_test_CV[i].reshape(64,64,64)})
#             io.savemat(path+'X_recon_highest'+'.mat',mdict={'X_recon_highest': X_nrmse.reshape(64,64,64)})
#             io.savemat(path+'diff_highest'+'.mat',mdict={'diff_highest': diff_vae_enc_abs[i].reshape(64,64,64)})
#             y_nrmse_highest = np.argwhere(np.all((Y_norm-Y_test_CV[i])==0, axis=1))
#             r_sq_highest = calc_R_sq(X_nrmse,X_test_CV[i])
#         if abs(NRMSE-nrmse_avg)<eps:
#             eps = abs(NRMSE-nrmse_avg)
#             io.savemat(path+'X_GT_avg'+'.mat',mdict={'X_GT_avg': X_test_CV[i].reshape(64,64,64)})
#             io.savemat(path+'X_recon_avg'+'.mat',mdict={'X_recon_avg': X_nrmse.reshape(64,64,64)})
#             io.savemat(path+'diff_avg'+'.mat',mdict={'diff_avg': diff_vae_enc_abs[i].reshape(64,64,64)})
#             y_nrmse_avg = np.argwhere(np.all((Y_norm-Y_test_CV[i])==0, axis=1))
#             r_sq_avg = calc_R_sq(X_nrmse,X_test_CV[i])
        
    MEP_av_t,MEP_act_av_t,MEP_var_t,MEP_act_var_t,Musc_Act_t = plot_assist(Y_test_CV,RMSE_all,Y)    
    
    MEP_av += MEP_av_t
    MEP_active_av += MEP_act_av_t
    MEP_var += MEP_var_t
    MEP_active_var += MEP_act_var_t
    Muscles_Active += Musc_Act_t
    rmse_all += RMSE_all
    nrmse_all += NRMSE_all

box_rmse, box_nrmse = plot_box(Y_index,Y_train,rmse_all,nrmse_all)


# print(nrmse_lowest,nrmse_avg,nrmse_highest)
# print(r_sq_lowest,r_sq_avg,r_sq_highest)
# print(y_nrmse_lowest,y_nrmse_avg,y_nrmse_highest)
# print(Y_norm.shape, Y_train.shape)
# print(Y_norm)

# list_int = [300,601,602,603,604,605,606,907,1021]
# list_stim = [i+112 for i in list_int]
# y_index_greater_count = [i for i in list_int]
# y_samp_index = [for ]

print(r_sq_all)
print(nrmse_all)

# %% [markdown]
# # [Test cell] above & [calc_nrmse_r_sq Func] below

# %%
# for choice_1 in [1,2]:
for choice_1 in [2]:
    # for choice_2 in [1,2,3,4]:  
    for choice_2 in [1,3]:     
        
        if choice_1 == 1:
            y_chosen=rmse_all
            box_data = box_rmse
            y_label = 'RMSE'
        else:
            y_chosen=nrmse_all
            box_data = box_nrmse
            y_label = 'NRMSE'

        if choice_2 == 1:
            color = MEP_active_av
            legend_label = 'Normalized MEP mean of active muscles'    
        elif choice_2 == 2:
            color = MEP_av
            legend_label = 'Normalized MEP mean of all muscles'
        elif choice_2 == 3:
            color = MEP_active_var
            legend_label = 'Normalized MEP variance of active muscles'
        else:
            color = MEP_var
            legend_label = 'Normalized MEP variance of all muscles'

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            x=Muscles_Active,
            y=y_chosen,  
            c=color,
#             y=color,  
#             c=y_chosen,
            cmap='copper') # other options are 'tab10', 'Greys'

        ax.set_xlabel('Number of active muscles')
        ax.set_ylabel(y_label)
        cbar = plt.colorbar(scatter)
        cbar.set_label(legend_label)

        ax.boxplot(box_data)

        plt.tight_layout()
        # path = "../../../Dropbox/Summer \'20/TMS_SD/TNSRE/May_10/"
        # plt.savefig(path+y_label+"-vs-Muscles-Active-"+legend_label+"-sub-3-all-folds.png")
        path = "../../../Dropbox/Repo_2022/TMS/Figs_TNSRE/"
        plt.savefig(path+y_label+"-vs-Muscles-Active-"+legend_label+"-sub-3-all-folds-v3.png")

# %% [markdown]
# # [Error vs. No of active muscles] above & [find_RMT Func] below

# %%
for choice_1 in [2]:
# for choice_1 in [1,2]:
    for choice_2 in [2,4]:    
    # for choice_2 in [1,2,3,4]:     
        
        if choice_1 == 1:
            y_chosen=rmse_all
            box_data = box_rmse
            y_label = 'RMSE'
        else:
            y_chosen=nrmse_all
            box_data = box_nrmse
            y_label = 'NRMSE'

        if choice_2 == 1:
            x_chosen = MEP_active_av
            legend_label = 'Normalized MEP mean of active muscles'    
        elif choice_2 == 2:
            x_chosen = MEP_av
            legend_label = 'Normalized MEP mean of all muscles'
        elif choice_2 == 3:
            x_chosen = MEP_active_var
            legend_label = 'Normalized MEP variance of active muscles'
        else:
            x_chosen = MEP_var
            legend_label = 'Normalized MEP variance of all muscles'

        fig, ax = plt.subplots()
        scatter = ax.scatter(
            x=x_chosen, 
            y=y_chosen, 
            c=RMT_level, 
            cmap='copper')
            # cmap='copper')
        #     cmap='autumn')

        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper right", title="% RMT")
        ax.add_artist(legend1)
        ax.set_xlabel(legend_label)
        ax.set_ylabel(y_label)
        # plt.ylim([0, 0.5])

        # plt.grid('True')    
        # plt.gca().set_aspect("equal")
        plt.tight_layout()
        # path = "../../../Dropbox/Summer \'20/TMS_SD/TNSRE/May_26/"
        path = "../../../Dropbox/Repo_2022/TMS/Figs_TNSRE/"
        plt.savefig(path+y_label+"-vs-"+legend_label+"-sub-3-all-folds-v2.png")

# %% [markdown]
# # [MEP Intensity vs. Error] above & [RMT_level_stats Func] below
