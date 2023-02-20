# %% Import libraries and set seeds
from utils import *

import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy import io
tf.config.list_physical_devices('GPU')
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
#from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.python.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
# import sklearn as sk

tf.keras.backend.clear_session()
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
tf.compat.v1.disable_eager_execution()
init_op = tf.compat.v1.global_variables_initializer()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
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

#%% Model [Direct Variational] Train below
folds = StratifiedKFold(n_splits=10, shuffle=True, 
                        random_state=0).split(Y_train, np.zeros(shape=(Y_train.shape[0],1)))
nrmse_all, r_sq_all, rmt_level = [], [], [] 

for j, (train_idx, test_idx) in enumerate(folds):

    if j>0:
        continue
    else:
        print(f'\n Subject: {args.sub}, Running Fold: {j+1}',)
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
        map_rev_vae.load_weights('weights/'+run_str_map+'.h5', by_name=True)
        decoded_imgs_vae_enc = map_rev_vae.predict(Y_test_CV)        
        diff_vae_enc_abs = abs(X_test_CV - decoded_imgs_vae_enc)       
        
        for i in range(Y_test_CV.shape[0]):
            X_nrmse = decoded_imgs_vae_enc[i]   
            nrmse, r_sq, _ = calc_nrmse_r_sq(X_nrmse,X_test_CV[i])
            nrmse_all.append(nrmse), r_sq_all.append(r_sq)   

        print('NRMSE_mean: %.3f' %(np.mean(nrmse_all)))
        print('NRMSE_std: %.3f' %(np.std(nrmse_all)))
        print('R-squared_mean: %.3f' %(np.mean(r_sq_all)))
        print('R-squared_std: %.3f' %(np.std(r_sq_all)))
