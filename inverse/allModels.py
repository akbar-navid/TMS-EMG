# %%
from utils import *
import numpy as np
import math
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import preprocessing
from itertools import product
from scipy import io
from scipy import stats
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Add, Input, Dense, Activation, Conv3D, MaxPooling3D, AveragePooling3D,concatenate, BatchNormalization, Dropout, Flatten, Reshape, Conv3DTranspose, UpSampling3D
from tensorflow.keras.models import Model, load_model
#from keras.utils import to_categorical
from tensorflow.keras.initializers import glorot_uniform, he_uniform
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.regularizers import l1, l2, l1_l2

from tensorflow.python.keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from contextlib import redirect_stdout
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.utils import normalize
from contextlib import redirect_stdout
from tensorflow.keras.activations import relu
from numpy import dot, inner
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

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
   

# %%
# import sklearn as sk
import matplotlib as mpl
print(mpl.__version__)

# %% [markdown]
# # [Intialization] above & [Data load: Sub-1] below

# %%
# Seed Setup
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

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

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

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

# %% [markdown]
# # [Data load: Sub-2] above & [Data load: Sub-3] below

# %%
# Seed Setup
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

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
print(list2)

while calc <= q:
    if calc in list2:
        calc += 1  
        continue
    else:
        list3.append(calc)
        calc += 1   
    
print(len(list3))

for i in list3:     
    if i<1000:
        tmp_mat = io.loadmat('data_026/E-box/026_E-box_0{0}.mat'.format(i)) 
        X[m,:,:,:,0] = np.array(tmp_mat['box_matrix_E'], dtype=data_type)     
        m+=1   
    else:
        tmp_mat = io.loadmat('data_026/E-box/026_E-box_{0}.mat'.format(i)) 
        X[m,:,:,:,0] = np.array(tmp_mat['box_matrix_E'], dtype=data_type)     
        m+=1       

# Load MEPs (Sub-MD, All RMTs)
tmp_mat = io.loadmat('data_026/Y_026.mat') 
Y = np.array(tmp_mat['MEPampNormAll'], dtype=data_type)  
print('Y:',Y.shape)
    
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

# print(a,b)
print(np.amax(X_train),np.amin(X_train))

print ("X shape: " + str(X_train.shape))
print ("Y shape: " + str(Y_train.shape))

# Load Mask
tmp_mat = io.loadmat('data_026//mask_026.mat')
motor_mask = tmp_mat['logical_mask']
motor_mask = motor_mask.reshape(1,*img_shape,1)
motor_mask = tf.convert_to_tensor(motor_mask, np.float32)

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
 
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):

    if j==0:   
        print(test_idx)    

# Y_nz_pos = np.any(Y_norm >= 1e-3, axis=1) 
Y_nz_pos = np.where(Y_ind)
# (array([1, 2], dtype=int64),)
# print(len(Y_ind))
# print(len(Y_nz_pos))
print(Y_nz_pos)
# np.argwhere(a>4)
Y_all_RMT_ind = Y_nz_pos[0]
print(Y_all_RMT_ind[test_idx])
# 20, 8, 10, 8
print(len(test_idx))
print(len(Y_train))

# -----------------------------------------

a = [1,2]
b = []
for i in a:
    b.append(i+5)
print(b)

# %% [markdown]
# # [Test cell] above & [VAE func] below


# %% [markdown]
# # [AE Train+Eval] below

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
 
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):

    if j!=2:
        continue
    else:
        print('\n Running Fold: ', j+1)
        X_train_CV = X_train[train_idx]  

        img = Input(shape=(64,64,64,1))
#         img_ae = AE(main_input=img, l_1=1e-4, bn=False, act_f=relu_out)
        img_ae = AE(main_input=img, l_1=1e-4, bn=True, act_f=relu_out)
        
        ae = None
        ae = Model(img, img_ae)    
        
#         run_str_ae = '026,AE,Feb_17,fold_'+str(j+1)
        run_str_ae = '026,AE,bn_T,Feb_17,fold_'+str(j+1)
        
        ae.compile(optimizer=Adadelta(lr=1.0), loss='mean_squared_error', metrics=['mse'])
        ae.summary()

        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=20, min_delta=1e-5) 
        lred = ReduceLROnPlateau(monitor='loss', factor=0.7,min_delta = 1e-5,
                                 mode='auto', patience=5,verbose=1, min_lr=1e-5, cooldown=0)
        mc = ModelCheckpoint('weights//'+run_str_ae+'.h5', monitor='loss', mode='auto',
                             save_best_only=True, save_weights_only=True, verbose=1)

        history = ae.fit(X_train_CV, X_train_CV, validation_split=0.0,
                         callbacks=[es,lred,mc], epochs=300, batch_size=16)
        
        ae.load_weights('weights//'+run_str_ae+'.h5', by_name=True)
        decoded_imgs_ae = ae.predict(X_test_CV)        
        diff_ae_abs = abs(X_test_CV - decoded_imgs_ae)     


# %% [markdown]
# # [Inv_Map_AE] func above & [Inv_Map_AE Train+Eval] below

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
 
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):

    if j>0:
        continue
    else:
        print('\n Running Fold: ', j+1) 
        X_train_CV = X_train[train_idx]  
        Y_train_CV = Y_train[train_idx]  
        X_test_CV = X_train[test_idx]  
        Y_test_CV = Y_train[test_idx]     

##        Sub-1    
#         run_str_ae = 'AE,s_MD,nz,n_stim,bn,relu,relu_out,Oct_28,fold_'+str(j+1)
#         run_str_map = 'MD,Conv,bn_T,May_30,fold_'+str(j+1)
#         run_str_map = 'MD,AE-dec,bn_T,May_30,' + run_str_ae   

##        Sub-2
        run_str_ae = '025,AE,bn_T,Mar_08,fold_'+str(j+1)
#         run_str_map = '025,Conv,bn_T,May_31,fold_' + str(j+1)
        run_str_map = '025,AE-dec,bn_T,May_31,fold_' + run_str_ae

##        Sub-3
#         run_str_ae = '026,AE,bn_T,Feb_17,fold_'+str(j+1)    
#         run_str_map = '026,Conv,bn_T,May_31,fold_' + str(j+1)
#         run_str_map = '026,AE-dec,bn_T,May_31,' + run_str_ae    

        inputs = Input(shape=MEP_shape)
        map_rev_ae = None        
        map_rev_ae = Inv_Map(inputs,l_1=1e-4,act_f=relu_out,bn=True)
        
        map_rev_ae.load_weights('weights//'+run_str_ae+'.h5', by_name=True)       
        for layer in map_rev_ae.layers[13:]:    # ---Streamlined Mapper, Conv, Enc, bn_T---#
            layer.trainable = False   

        for layer in map_rev_ae.layers:
            print(layer, layer.trainable)
        
        map_rev_ae.compile(optimizer=Adadelta(lr=1.0), loss='mean_squared_error', metrics=['mse'])
        map_rev_ae.summary()

        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=20, min_delta=1e-5) 
        lred = ReduceLROnPlateau(monitor='loss', factor=0.7, min_delta = 1e-5,
                                 mode='auto', patience=5, verbose=1, min_lr=1e-5, cooldown=0)
        mc = ModelCheckpoint('weights//'+run_str_map+'.h5', monitor='loss', mode='auto',
                             save_best_only=True, save_weights_only=True, verbose=1)

        history = map_rev_ae.fit(Y_train_CV, X_train_CV, validation_split=0.0, 
                                 callbacks=[es,lred,mc], epochs=300, batch_size=16)
#                                  callbacks=[es,lred], epochs=10, batch_size=16)
        
        map_rev_ae.load_weights('weights//'+run_str_map+'.h5', by_name=True)    
        decoded_imgs_ae = map_rev_ae.predict(Y_test_CV, batch_size=16)       
        diff_ae = abs(X_test_CV - decoded_imgs_ae)

        NRMSE_all = []
        RMSE_all = []
        R_sq_all = []
        
        for i in range(Y_test_CV.shape[0]):
            X_nrmse = decoded_imgs_ae[i]   
            nrmse, r_sq, _ = calc_nrmse_r_sq(X_nrmse,X_test_CV[i])
            NRMSE_all.append(nrmse)
            R_sq_all.append(r_sq)

        print('NRMSE_mean: ', np.mean(NRMSE_all))
        print('NRMSE_SEM: ', 1.96*stats.sem(NRMSE_all,ddof=0))
        print('R-squared_mean: ', np.mean(R_sq_all))
        print('R-squared_SEM: ', 1.96*stats.sem(R_sq_all,ddof=0))    

# %%
mu_AE = np.mean(AE)
# mu_VAE_enc = np.mean(VAE_L1)
mu_VAE_samp = np.mean(VAE_Samp_L2)
mu_VAE_L1_no_0 = np.mean(VAE_L1_no_0)

std_AE = np.std(AE)
# std_VAE_enc = np.std(VAE_L1)
std_VAE_samp = np.std(VAE_Samp_L2)
std_VAE_L1_no_0 = np.std(VAE_L1_no_0)

# labels = ['VAE [without zero activs]', 'VAE [with zero activs]', 'AE [without zero activs]']
labels = ['VAE_enc', 'VAE_samp', 'AE']
x_pos = np.arange(len(labels))

# CTEs = [mu_AE, mu_VAE_enc, mu_VAE_samp]
# error = [std_AE, std_VAE_enc, std_VAE_samp]
CTEs = [mu_VAE_L1_no_0, mu_VAE_samp,mu_AE]
error = [std_VAE_L1_no_0, std_VAE_samp,std_AE]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_title('E-field recon performances: Sub_025 [without zero activations]')
ax.yaxis.grid(True)
axes = plt.gca()
axes.set_ylim([0,0.4])

plt.tight_layout()
plt.savefig('images//E-field recon performances for Subject_025_non_zero_July_22.png')
# ,ddof = 1

# %% [markdown]
# # [Performance Plots] above & [VAE Train+Eval] below

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
 
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):
    
    if j>10:
        continue
    else:
        print('\n Running Fold: ', j+1)
        X_train_CV = X_train[train_idx]  
        Y_train_CV = Y_train[train_idx]  
        X_test_CV = X_train[test_idx]  
        Y_test_CV = Y_train[test_idx] 

        img = Input(shape=(64,64,64,1))
        img_vae, loss_vae = VAE(main_input=img, l_1=1e-4, act='relu', act_f=relu_out)
    
        vae = None
        vae = Model(img, img_vae) 

        run_str_vae = '026,VAE,l1,Feb_16,fold_'+str(j+1)
    
        vae.compile(optimizer=Adadelta(lr=1.0), loss=loss_vae, metrics=['mse'])
        vae.summary()  
        
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=20, min_delta=1e-5) 
        lred = ReduceLROnPlateau(monitor='loss', factor=0.7,min_delta = 1e-5,
                                 mode='auto', patience=5,verbose=1, min_lr=1e-5, cooldown=0)
        mc = ModelCheckpoint('weights//'+run_str_vae+'.h5', monitor='loss', mode='auto',
                             save_best_only=True, save_weights_only=True, verbose=1)

        history = vae.fit(X_train_CV, X_train_CV, validation_split=0.,
                         callbacks=[es,lred,mc], epochs=300, batch_size=16)
        
        vae.load_weights('weights//'+run_str_vae+'.h5', by_name=True)
        decoded_imgs_vae = vae.predict(X_test_CV)        
        diff_vae_abs = abs(X_test_CV - decoded_imgs_vae)     

        mse = np.mean((X_test_CV - decoded_imgs_vae)**2)
        print('The mse is: %.6f' %mse)
        img_pow = np.mean(X_test_CV**2)
        print('The img_pow is: %.6f' %img_pow)
        NRMSE = np.sqrt(mse/img_pow)
        print('The normalized rmse is: %.6f' %NRMSE)         


# %% [markdown]
# # [Inv_Map_VAE] func above & [Inv_Map_VAE Train+Eval] func below

# %%
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)

nrmse_all, r_sq_all, rmt_level = [], [], [] 
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))

for j, (train_idx, test_idx) in enumerate(folds):

    if j>9:
        continue
    else:
        print('\n Running Fold: ', j+1)
        X_train_CV, Y_train_CV = X_train[train_idx], Y_train[train_idx]  
        X_test_CV, Y_test_CV = X_train[test_idx], Y_train[test_idx]  
                   
        mep = Input(shape=MEP_shape)
        map_rev_vae = None
        
        img_map, loss_map = Inv_Map_VAE_Lat_2d(inputs=mep,l_1=1e-4,act_f=relu_out) 
#         img_map, loss_map = Inv_Map_VAE_Lat_2d(inputs=mep, l_1=1e-4, act_f=relu_out, vae_std=True) 

#         img_map = Inv_Map_VAE_Lat_2d(inputs=mep, l_1=1e-4, act_f=relu_out, vae_dec=True) 
#         loss_map = 'mean_squared_error'
    
##        Sub-1
#         run_str_vae = 'VAE,s_MD,nz,n_stim,b_T,Aug_24,fold_'+str(j+1)    
#         run_str_map = 'Inv_Map_Lat,enc_map,no_weights,l1,Sep_29,fold_'+str(j+1)
#         run_str_map = 'Inv_Map_Lat,enc_map,l1,Oct_27,' + run_str_vae
#         run_str_map = 'Map,enc,l1,b_T,Aug_24' + run_str_vae
    
##        Sub-2
#         run_str_vae = '025,VAE,l1,Mar_03,fold_'+str(j+1)
        run_str_map = '025,Variational,l1,Oct_01,fold_'+str(j+1)
#         run_str_map = '025,VAE-std,l1,Mar_03,fold_'+str(j+1) 
#         run_str_map = '025,VAE-dec,l1,Mar_08,fold_'+str(j+1)
    
##        Sub-3
#         run_str_vae = '026,VAE,l1,Feb_16,fold_'+str(j+1)
#         run_str_map = '026,Variational,no_weights,l1,Jan_20,fold_'+str(j+1)
#         run_str_map = '026,VAE-mod,l1,Feb_16,' + run_str_vae    
#         run_str_map = '026,VAE-std,l1,Feb_16,' + run_str_vae   
#         run_str_map = '026,VAE-std-old,l1,Feb_16,' + run_str_vae   

        map_rev_vae = Model(mep, img_map)      
#         map_rev_vae.load_weights('weights//'+run_str_vae+'.h5', by_name=True)

#         for layer in map_rev_vae.layers[4:]:    # ---Vae_Std (New/Old)---#
# #         for layer in map_rev_vae.layers[6:]:    # ---Vae_Mod---#
#             layer.trainable = False            
#         for layer in map_rev_vae.layers:
#             print(layer, layer.trainable)
        
        map_rev_vae.compile(optimizer=Adadelta(lr=1.0), loss=loss_map, metrics=['mse'])
#         map_rev_vae.summary()   
        es = EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=20, min_delta=1e-5) 
        lred = ReduceLROnPlateau(monitor='loss', factor=0.7,min_delta = 1e-5,
                                 mode='auto', patience=5,verbose=1, min_lr=1e-5, cooldown=0)
        mc = ModelCheckpoint('weights//'+run_str_map+'.h5', monitor='loss', mode='auto',
                             save_best_only=True, save_weights_only=True, verbose=1)

        history = map_rev_vae.fit(Y_train_CV, X_train_CV, validation_split=0., 
                                 callbacks=[es,lred,mc], epochs=300, batch_size=16)      
        
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
# # [Test Cell] above & [Temp: Sub-1] below

# %%
    
# AE-Dec, Fold-1
NRMSE_mean:  0.28819305
NRMSE_SEM:  0.0464238272388398
R-squared_mean:  0.8340684087640038
R-squared_SEM:  0.04678401054660773
    
# Conv, Fold-1
NRMSE_mean:  0.21381421
NRMSE_SEM:  0.035396586201048456
R-squared_mean:  0.8301662730330694
R-squared_SEM:  0.046834976499224606

# VAE-Dec, Fold-1
NRMSE_mean:  0.20336421
NRMSE_SEM:  0.036425354124237606
R-squared_mean:  0.8494466995086704
R-squared_SEM:  0.047234967437579205
    
# VAE-Samp-Dec, Fold-1
NRMSE_mean:  0.1696584
NRMSE_SEM:  0.032549567839310414
R-squared_mean:  0.8627302197698643
R-squared_SEM:  0.043382202455972844
    
# Mar_03, Variational, Fold-1  
NRMSE_mean:  0.16707353
NRMSE_SEM:  0.031174588421354246
R-squared_mean:  0.8666947496676968
R-squared_SEM:  0.04275602381179072

Direct Variational & \textbf{0.171 ± 0.008} & 0.869 ± 0.010
0.174 ± 0.008 & 0.865 ± 0.010
0.208 ± 0.010 & 0.852 ± 0.010
0.219 ± 0.011 &  0.832 ± 0.010
0.294 ± 0.014 & 0.836 ± 0.011

For RMT- 110
NRMSE_mean:  0.18970938
NRMSE_SEM:  0.06089570647814189
R-squared_mean:  0.8509547222285847
R-squared_SEM:  0.07435319493705407
For RMT- 120
NRMSE_mean:  0.1699419
NRMSE_SEM:  0.05949641040935783
R-squared_mean:  0.8215869132515874
R-squared_SEM:  0.12034987914599511
For RMT- 130
NRMSE_mean:  0.13615103
NRMSE_SEM:  0.03393565587794361
R-squared_mean:  0.9231356226564071
R-squared_SEM:  0.027908492759400858
For RMT- 140
NRMSE_mean:  0.15682876
NRMSE_SEM:  0.0718935264718336
R-squared_mean:  0.8645238417686311
R-squared_SEM:  0.11009353844599706

# %%
    
# AE-Dec, Fold-1
NRMSE_mean:  0.39833382
NRMSE_SEM:  0.03431423367219637
R-squared_mean:  0.6937159757202419
R-squared_SEM:  0.04237899604187923
    
# Conv, Fold-1
NRMSE_mean:  0.2499095
NRMSE_SEM:  0.019230893056503377
R-squared_mean:  0.7220448257470738
R-squared_SEM:  0.04403264289051753
    
# VAE-Dec, Fold-1
NRMSE_mean:  0.24268302
NRMSE_SEM:  0.020036021739937987
R-squared_mean:  0.7459908970394428
R-squared_SEM:  0.044900072206229824
    
# VAE-Samp-Dec, Fold-1
NRMSE_mean:  0.21760412
NRMSE_SEM:  0.01780095575686287
R-squared_mean:  0.7659321839094199
R-squared_SEM:  0.04455070704143456
    
# Mar_03, Variational, Fold-1
NRMSE_mean:  0.20897153
NRMSE_SEM:  0.01754533451136653
R-squared_mean:  0.7671335112761444
R-squared_SEM:  0.044853316316713965    

Direct Variational & \textbf{0.211 ± 0.006} & \textbf{0.765 ± 0.010}
0.220 ± 0.005 & 0.764 ± 0.010
0.245 ± 0.006 & 0.744 ± 0.010
0.252 ± 0.007 & 0.720 ± 0.009
0.402 ± 0.011 & 0.692 ± 0.009

For RMT- 110
NRMSE_mean:  0.22917397
NRMSE_SEM:  0.042778590230099
R-squared_mean:  0.793114373725593
R-squared_SEM:  0.0845550273793225
For RMT- 120
NRMSE_mean:  0.19102418
NRMSE_SEM:  0.023093285417817046
R-squared_mean:  0.7945676014189591
R-squared_SEM:  0.07572953206699508
For RMT- 130
NRMSE_mean:  0.21012439
NRMSE_SEM:  0.03923163108673558
R-squared_mean:  0.7667460591939294
R-squared_SEM:  0.09927818317870425
For RMT- 140
NRMSE_mean:  0.20763187
NRMSE_SEM:  0.03345135614833034
R-squared_mean:  0.7148347325191606
R-squared_SEM:  0.09661803580740252

# %% [markdown]
# # [Temp: Sub-2] above & [Temp: Sub-3] below

# %%
    
# AE-Dec, Fold-3
NRMSE_mean:  0.23667833
NRMSE_SEM:  0.026138220409968972
R-squared_mean:  0.5950051519746618
R-squared_SEM:  0.0480405343269679
    
# Conv, Fold-3
NRMSE_mean:  0.2100306
NRMSE_SEM:  0.021676616959231194
R-squared_mean:  0.5997294140213695
R-squared_SEM:  0.048635722636942845
    
# VAE-Dec
NRMSE_mean:  0.2046505
NRMSE_SEM:  0.022177138092841504
R-squared_mean:  0.6253453796198464
R-squared_SEM:  0.050223258995498093 

# VAE-Samp-Dec   
NRMSE_mean:  0.15762916
NRMSE_SEM:  0.017954835733346046
R-squared_mean:  0.7163631074343878
R-squared_SEM:  0.04747357229976747

# Variational, Fold-3
NRMSE_mean:  0.14882809
NRMSE_SEM:  0.017540578946356335
R-squared_mean:  0.7448044647484566
R-squared_SEM:  0.0455029762117444

Direct Variational & \textbf{0.157 ± 0.005} & \textbf{0.709 ± 0.015}
0.166 ± 0.005 & 0.682 ± 0.014
0.216 ± 0.007 & 0.595 ± 0.013
0.222 ± 0.007 & 0.571 ± 0.012
0.250 ± 0.008 & 0.566 ± 0.012

For RMT- 110
NRMSE_mean:  0.1579761
NRMSE_SEM:  0.039542473422652755
R-squared_mean:  0.7180003601265715
R-squared_SEM:  0.10462672782270811
For RMT- 120
NRMSE_mean:  0.1659879
NRMSE_SEM:  0.03791114263772722
R-squared_mean:  0.6954750378746
R-squared_SEM:  0.0815076130955896
For RMT- 130
NRMSE_mean:  0.12715267
NRMSE_SEM:  0.0290974844757038
R-squared_mean:  0.7787031709615919
R-squared_SEM:  0.09019668889986877
For RMT- 140
NRMSE_mean:  0.13678285
NRMSE_SEM:  0.022442696705156535
R-squared_mean:  0.8072407535434244
R-squared_SEM:  0.08282853957005877

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
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)    

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
# # [CV Stim No.] above & [Plot Assist Func] below


# %% [markdown]
# # [Plot Box Func] above & [Metrics Across Folds] below

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

# %%
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

# %% [markdown]
# # [Dump] below

# %%
# # Extending the dimension of MEP_av to include RMT labels    
# MEP_av = np.asarray(MEP_av)
# MEP_av = np.expand_dims(MEP_av, axis=1)
# new_col = MEP_av.sum(1)[...,None] # None keeps (n, 1) shape
# MEP_av = np.concatenate((MEP_av, new_col), 1)
# print(MEP_av.shape) 

# # RMT label ranges found from above cell   
# # 0-22,23-62,63-88,89-
# MEP_av[0:23,1] = 110
# MEP_av[23:63,1] = 120
# MEP_av[63:89,1] = 130
# MEP_av[89:,1] = 140

# m, b = np.polyfit(MEP_av[:,0], NRMSE_all, 1)
# plt.plot(MEP_av[:,0], m*MEP_av[:,0] + b)

# %%
# VAE Error Histogram

h = X_test_CV-decoded_imgs_vae
rec_err = h.flatten()
bins = 80
plt.hist(rec_err,bins=bins)
plt.yscale('log', nonposy='clip')
axes = plt.gca()
axes.set_ylim([1e1,2e7])
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.title("Test recon errors (VAE,hid_lay=16,map=encoded,L2)")
ax = plt.gca()
ax.set_facecolor('xkcd:grey')
plt.savefig('images//Test recon errors (VAE,hid_lay=16,map=encoded,L2)'+'.png',dpi=300, bbox_inches='tight')

# %%
# AE Error Histogram

h = X_test_CV-decoded_imgs_ae
rec_err = h.flatten()
bins = 80

plt.hist(rec_err,bins=bins)
plt.yscale('log', nonposy='clip')
axes = plt.gca()
axes.set_ylim([1e1,2e7])
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.title("Test recon errors (AE,act=relu_max)")
ax = plt.gca()
ax.set_facecolor('xkcd:grey')
plt.savefig('images//Test recon errors (AE,act_f=relu_max).png',dpi=300, bbox_inches='tight')

# %%
#VAE Recon Images

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

num = 3 #Number of pictures to show
fig = plt.figure(figsize=(24,24))

for i in range(num):
#     idx = np.random.randint(1,X_test.shape[0])
    idx=i+30
    
    # GT
    ax = fig.add_subplot(3,num,i+1, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)
    surf = ax.plot_surface(x,y,np.sum(X_test_CV[idx].reshape(64,64,64),2), cmap=plt.cm.jet, 
                           antialiased=True, vmin=0, vmax=16);
    fig.colorbar(surf, shrink=0.5)   
    plt.title('Ground Truth, Trial='+str(idx))
    plt.axis('off');
    
    # Decoded
    ax = fig.add_subplot(3,num,i+1+num, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)
    surf = ax.plot_surface(x,y,np.sum(decoded_imgs_vae[idx].reshape(64,64,64),2), cmap=plt.cm.jet,  
                           antialiased=True, vmin=0, vmax=16)  
    fig.colorbar(surf, shrink=0.5)   
    plt.title('VAE Recon o/p, Trial='+str(idx))
    plt.axis('off');    
    
    # Error
    ax = fig.add_subplot(3,num,i+1+num*2, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)   
    surf = ax.plot_surface(x,y,np.sum((abs(X_test_CV[idx]-decoded_imgs_vae[idx])).reshape(64,64,64),2), cmap=plt.cm.jet,  
                           antialiased=True, vmin=0, vmax=16)    
#     surf.set_clim(0,15)
    fig.colorbar(surf, shrink=0.5)  
    plt.title('Absolute Error, Trial='+str(idx))
    plt.axis('off');

plt.savefig('images//VAE_abs_error_hid_lay=16,map=encoded,L2'+run_str_vae+'_v3.png',dpi=300, bbox_inches='tight')

# %%
#AE Recon Images

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

num = 3 #Number of pictures to show
fig = plt.figure(figsize=(24,24))

for i in range(num):
#     idx = np.random.randint(1,X_test.shape[0])
    idx=i+15
    
    # GT
    ax = fig.add_subplot(3,num,i+1, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)
    surf = ax.plot_surface(x,y,np.sum(X_test_CV[idx].reshape(64,64,64),2), cmap=plt.cm.jet, 
                           antialiased=True, vmin=0, vmax=16);
    fig.colorbar(surf, shrink=0.5)   
    plt.title('GT Test, Trial='+str(idx))
    plt.axis('off');
    
    # Decoded 
    ax = fig.add_subplot(3,num,i+1+num, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)
    surf = ax.plot_surface(x,y,np.sum(decoded_imgs_ae[idx].reshape(64,64,64),2), cmap=plt.cm.jet,  
                           antialiased=True, vmin=0, vmax=16)  
    fig.colorbar(surf, shrink=0.5)   
    plt.title('AE Recon o/p, Trial='+str(idx))
    plt.axis('off');    
    
    # Error
    ax = fig.add_subplot(3,num,i+1+num*2, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)    
    surf = ax.plot_surface(x,y,np.sum((abs(X_test_CV[idx]-decoded_imgs_ae[idx])).reshape(64,64,64),2), cmap=plt.cm.jet,  
                           antialiased=True, vmin=0, vmax=16)    
#     surf.set_clim(0,15)
    fig.colorbar(surf, shrink=0.5)  
    plt.title('Absolute Error, Trial='+str(idx))
    plt.axis('off');

plt.savefig('images//AE,act_f=relu_max'+run_str_ae+'_v3.png',dpi=300, bbox_inches='tight')


