# %%
import numpy as np
import math
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn import preprocessing
from itertools import product
from scipy import io
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Lambda, Add, Input, Dense, Activation, Conv3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D, concatenate, BatchNormalization, Dropout, Flatten
from keras.models import Model, load_model
#from keras.utils import to_categorical
from keras.initializers import glorot_uniform, he_uniform
from keras.optimizers import Adam, SGD, Adagrad, Adadelta
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.regularizers import l1, l2, l1_l2
from tensorflow.losses import huber_loss

from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import normalize

%matplotlib inline
from astropy.stats import jackknife_stats


# %%
# Loading the dataset
encoded_train = np.load('Python_Data//Encoded_All_RMT_16_Cube_Final_Train.npy').reshape(629,16,16,16,1)   #Data
encoded_test  = np.load('Python_Data//Encoded_All_RMT_16_Cube_Final_Test.npy').reshape(70,16,16,16,1)   #Data
# encoded_train = np.load('2_Masked_Firing_Norm_Python_Run_3//Encoded_All_RMT_Firing_v3_Train.npy').reshape(629,16,16,16,1)   #Data
# encoded_test  = np.load('2_Masked_Firing_Norm_Python_Run_3//Encoded_All_RMT_Firing_v3_Test.npy').reshape(70,16,16,16,1)   #Data
Y_train       = np.load('Python_Data//Y_All_RMT_Train.npy').reshape(629,15)
Y_test        = np.load('Python_Data//Y_All_RMT_Test.npy').reshape(70,15)
# Y_train       = np.load('2_Masked_Firing_Norm_Python_Run_3//Y_All_RMT_Train.npy').reshape(629,15)
# Y_test        = np.load('2_Masked_Firing_Norm_Python_Run_3//Y_All_RMT_Test.npy').reshape(70,15)

n_files          =  699

syn_data         = 'Weights_All_RMT_Stims'  
syn_var_name     = 'Weights_All_RMT_Stims'
ds               = 1
img_shape        = ((64-1)//ds+1, (64-1)//ds+1, (64-1)//ds+1) # 32 * 32 *32

data_type        = np.float32
tmp_syn          = io.loadmat('{0}.mat'.format(syn_data))
syn_weights      = np.array(tmp_syn[syn_var_name], dtype=data_type) 
synergies        = tf.convert_to_tensor(syn_weights, np.float32)

# Normalization
max_val = np.amax(encoded_train)
min_val = np.amin(encoded_train)
encoded_train = (encoded_train - min_val)/(max_val - min_val)
max_val = np.amax(encoded_train)
min_val = np.amin(encoded_train)
print("new max_val of encoded train set = " + str(max_val))
print("new min_val of encoded train set = " + str(min_val))

max_val = np.amax(encoded_test)
min_val = np.amin(encoded_test)
encoded_test = (encoded_test - min_val)/(max_val - min_val)
max_val = np.amax(encoded_test)
min_val = np.amin(encoded_test)
print("new max_val of encoded test set = " + str(max_val))
print("new min_val of encoded test set = " + str(min_val))

#Set split
X_train = encoded_train
X_test = encoded_test
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
print ("Weight Matrix shape: " + str(synergies.shape))

# %%
#Print several E-fields
num = 5 #Number of pictures to show
fig = plt.figure(figsize=(24,4))

for i in range(num):
    ax = fig.add_subplot(1,5,i+1, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,16,16,16)
    y = np.linspace(1,16,16,16)
    x, y = np.meshgrid(x,y)
    idx = np.random.randint(1,X_train.shape[0])
    ax.plot_surface(x,y,np.sum(X_train[idx].reshape(16,16,16),2), cmap=plt.cm.jet, antialiased=False);
    plt.title(idx)
    plt.axis('off');

# %%
def CNN_MEP(net, final='sigmoid', dr=False, dr_rate=0, l_1=0, l_2=0, activ = 'relu', bn=True,
            d1=128, d2=32, dim_red=True, input_shape=(16,16,16,1), filters=(16,32,64), init='glorot_uniform', bn_d=False):
    
    inputs = Input(shape=input_shape, name='input')
    activ1 = activ
    activ2 = activ
    activ2d = activ    
    
    #Phase 1
    x = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), kernel_size=(3,3,3), padding='same', kernel_initializer=init, name='conv_1')(inputs)
    if net == 'syn':
        x  = Activation('relu')(x)
    else:        
        x  = Activation(activ2)(x)
    if bn:
        x  = BatchNormalization()(x)
    x  = MaxPooling3D(padding='same', name='pool_1')(x)
    
    x = Conv3D(filters=filters[1], kernel_size=(3,3,3), kernel_regularizer=l1_l2(l1=l_1, l2=l_2), padding='same', kernel_initializer=init, name='conv_2')(x)
    if net == 'syn':
        x  = Activation('relu')(x)
    else:        
        x  = Activation(activ2)(x)
    if bn:
        x  = BatchNormalization()(x)
    x  = MaxPooling3D(padding='same', name='pool_2')(x)
    
    x = Conv3D(filters=filters[2], kernel_size=(3,3,3), kernel_regularizer=l1_l2(l1=l_1, l2=l_2), padding='same', kernel_initializer=init, name='conv_3')(x)
    if net == 'syn':
        x  = Activation('relu')(x)
    else:        
        x  = Activation(activ2)(x)
    if bn:
        x  = BatchNormalization()(x)
    if dim_red:
        x  = MaxPooling3D(padding='same', name='pool_3')(x)
        
    
    x = Flatten()(x)    
    
    #Phase 2
    x = Dense(units=d1, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), name='fc_1')(x)
    if net == 'syn':
        x  = Activation('relu')(x)   
    else:        
        x  = Activation(activ2d)(x)
        if dr:
            x = Dropout(dr_rate)(x) 

    x = Dense(units=d2, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), name='fc_2')(x)
    if net == 'syn':
        x  = Activation('relu')(x)  
    else:        
        x  = Activation(activ2d)(x)
        if dr:
            x = Dropout(dr_rate)(x) 
            

    if net == 'syn':
        x = Dense(units=9, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), name='syn')(x)
        x = Lambda(lambda x: tf.matmul(x,synergies))(x)    
#     final = Add()([MEP,SYN_to_MEP])  
        y = Activation('sigmoid')(x)
        if dr:
            x = Dropout(dr_rate)(x) 
    
    if net == 'res':
        x = Dense(units=15, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), name='res')(x)
#         x = Dense(units=15, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), name='res')(x)
        y = Activation(final)(x)
        if dr:
            x = Dropout(dr_rate)(x) 
        
    CNN_MEP = Model(inputs=inputs, outputs=y)
    
    return CNN_MEP

# %%
init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
sess.run(init_op)

# %%
mapper = CNN_MEP('syn',dim_red=False,d1=2000,d2=100)
# mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error', metrics=['accuracy'])

mapper.load_weights('Model_Summary//Mapper_Syn.h5')
w1, b1 = mapper.layers[22].get_weights()
print('Max weight is: ', str(np.amax(w1)))
print('Min weight is: ', str(np.amin(w1)))
print('Shape of weight matrix is: ', str(w1.shape))

h = w1
h[h==0]=float('nan')
fig = plt.figure(figsize=(9,3))
plt.hist(h)
plt.title('Weights of Dense Layer ' + str(w1.shape))
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig(fname='Model_Summary//Weights//Weights_of_Dense_Layer_'+str(w1.shape)+'_.png', dpi=200, bbox_inches='tight')
plt.show()

# %%
from contextlib import redirect_stdout
j='SYN_3_layer_Firing'
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1.0
    drop = 0.7
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        
    return lrate
callbacks = [LearningRateScheduler(step_decay)]

mapper = CNN_MEP('syn',dr=True,dr_rate=0.15,l_1=0.0001)
# mapper = CNN_MEP('syn',dr=False)
# mapper.load_weights('Model_Summary//ICASSP_Sims//Weights_'+j+'_.h5')
# mapper.load_weights('Model_Summary//Param_Check//Mapper_Fold_'+ str(7) + '.h5')
mapper.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')

# mapper.summary()
# with open('Model_Summary//ICASSP_Sims_Run_2//Summary_'+j+'_.txt', 'w') as f:
#     with redirect_stdout(f):
#         mapper.summary()
#Adadelta(lr=1)         mapper.compile(optimizer=Adagrad(lr=0.00001))

#Train validation_data=
#     callbacks = []
mapper.fit(X_train, Y_train, epochs=150, batch_size=16, callbacks=callbacks)
# mapper.save('Model_Summary//ICASSP_Sims//Weights_'+j+'_v2.h5')
mapper.save('Model_Summary//ICASSP_Sims_Run_3//Weights_'+j+'_.h5')

    #Validation
estim = mapper.predict(X_test, batch_size=16)
estim_train = mapper.evaluate(X_train, Y_train, batch_size=16)
estim_val = mapper.evaluate(X_test, Y_test, batch_size=16)
print('The train loss is: %.6f' %estim_train)
print('The test loss is: %.6f' %estim_val)
mse = np.mean((Y_test - estim)**2)
print('The mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(Y_test**2))
print('The normalized rmse is: %.6f' %NRMSE)    

# %%
mapper = CNN_MEP('syn',dr=True,dr_rate=0.15,l_1=0.0001)
mapper.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')
mapper.load_weights('Model_Summary//ICASSP_Sims//Weights_SYN_16_cube_3_layer_.h5')
# mapper.load_weights('Model_Summary//ICASSP_Sims_Run_3//Weights_SYN_3_layer_Firing_.h5')
# Y_test, Y_train, X_train, X_test
estim = mapper.predict(X_train, batch_size=16)
estim_train = mapper.evaluate(X_train, Y_train, batch_size=16)
print('The train loss is: %.6f' %estim_train)
mse = np.mean((Y_train - estim)**2)
print('The mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(Y_train**2))
print('The normalized rmse is: %.6f' %NRMSE)    

# %%
mapper = CNN_MEP('res',dr=True,dr_rate=0.15,l_1=0.0001)
mapper.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')
mapper.load_weights('Model_Summary//ICASSP_Sims_Run_3//Weights_MEP_3_layer_NF_.h5')

estim = mapper.predict(X_train, batch_size=16)
estim_train = mapper.evaluate(X_train, Y_train, batch_size=16)
print('The evaluated train loss is: %.6f' %estim_train)
mse = np.mean((Y_train - estim)**2)
print('The calcualated mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(Y_train**2))
print('The normalized rmse is: %.6f' %NRMSE)    

# %%
residual = CNN_MEP('res',final='tanh',activ='linear',dr=True,dr_rate=0.15,l_1=0.0001)
residual.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')
# residual.load_weights('Model_Summary//ICASSP_Sims_Run_3//Weights_Both_3_layer_Firing_v2_.h5')
residual.load_weights('Model_Summary//ICASSP_Sims//Weights_Both_16_cube_3_layer_v2_.h5')

    #Validation
p1 = mapper.predict(X_train, batch_size=16)
p2 = residual.predict(X_train, batch_size=16)
estim = p1 + p2    

mse = np.mean((Y_train - estim)**2)
print('The calcualated mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(Y_test**2))
print('The normalized rmse is: %.6f' %NRMSE)    

# %%
j='MEP_3_layer_NF'
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1.0
    drop = 0.7
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))        
    return lrate

callbacks = [LearningRateScheduler(step_decay)]

mapper = CNN_MEP('res',dr=True,dr_rate=0.15,l_1=0.0001)
# mapper.load_weights('Model_Summary//ICASSP_Sims//Weights_'+j+'_.h5')
# mapper.load_weights('Model_Summary//Param_Check//Mapper_Fold_'+ str(7) + '.h5')
mapper.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')

# mapper.summary()
# with open('Model_Summary//ICASSP_Sims_Run_2//Summary_'+j+'_.txt', 'w') as f:
#     with redirect_stdout(f):
#         mapper.summary()
#Adadelta(lr=1)         mapper.compile(optimizer=Adagrad(lr=0.00001))

#Train validation_data=
#     callbacks = []
mapper.fit(X_train, Y_train, epochs=150, batch_size=16, callbacks=callbacks)
# mapper.save('Model_Summary//ICASSP_Sims//Weights_'+j+'_v2.h5')
mapper.save('Model_Summary//ICASSP_Sims_Run_3//Weights_'+j+'_.h5')

    #Validation
estim = mapper.predict(X_test, batch_size=16)
estim_train = mapper.evaluate(X_train, Y_train, batch_size=16)
estim_val = mapper.evaluate(X_test, Y_test, batch_size=16)
print('The train loss is: %.6f' %estim_train)
print('The test loss is: %.6f' %estim_val)
mse = np.mean((Y_test - estim)**2)
print('The mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(Y_test**2))
print('The normalized rmse is: %.6f' %NRMSE)    

# %%
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(X_train, np.zeros(shape=(Y_train.shape[0], 1)))
from contextlib import redirect_stdout
        
for j, (train_idx, val_idx) in enumerate(folds):

    # Train and validation split
    print('\n Running Fold: ', j+1)
    X_trainCV = X_train[train_idx]
    Y_trainCV = Y_train[train_idx]
    X_valCV = X_train[val_idx]
    Y_valCV = Y_train[val_idx]    
 
    loss_fold = [] #List of the K different losses for every fold
        
        #Model 
    if j == 0:
        mapper = CNN_MEP('res',final='tanh',activ='relu',dr=True,dr_rate=0,l_1=0)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(4)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 1:
        mapper = CNN_MEP('res',final='tanh',activ='elu',dr=True,dr_rate=0,l_1=0)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(4)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 2:
        mapper = CNN_MEP('res',final='tanh',activ='elu',dr=True,dr_rate=0.15,l_1=0.0001)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(4)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 3:
        mapper = CNN_MEP('res',final='tanh',activ='elu',dr=True,dr_rate=0.1,l_1=0.0001)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(3)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    if j == 4:
        mapper = CNN_MEP('res',final='tanh',activ='linear',dr=True,dr_rate=0,l_1=0)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(4)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 5:
        mapper = CNN_MEP('res',final='tanh',activ='linear',dr=True,dr_rate=0.15,l_1=0.0001)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(4)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 6:
        mapper = CNN_MEP('res',final='tanh',activ='linear',dr=True,dr_rate=0.1,l_1=0.0001)
        mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(3)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 7:
        mapper = CNN_MEP('syn',dr=True,dr_rate=0,l_1=0)
#         mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(j)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    elif j == 8:
        mapper = CNN_MEP('syn',dr=True,dr_rate=0.15,l_1=0.0001)
#         mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(j)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
    else:        
        mapper = CNN_MEP('syn',dr=True,dr_rate=0.15,l_1=0.0001)
#         mapper.load_weights('Model_Summary//Dropout//Mapper_Reg_Fold_'+str(j)+'.h5')
        mapper.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error')
#         linear
#     mapper.summary()
    with open('Model_Summary//Param_Check_Mapper_New//Summary_Fold_'+str(j)+'_.txt', 'w') as f:
        with redirect_stdout(f):
            mapper.summary()
#         mapper.compile(optimizer=Adagrad(lr=0.00001), loss='mean_squared_error', metrics=['mae','mse'])

    #Train validation_data=
#     callbacks = []
    if j >= 7:
        mapper.fit(X_trainCV, Y_trainCV, validation_data=(X_valCV,Y_valCV), epochs=100, batch_size=16)
    else:
        mapper.fit(X_trainCV, -Y_trainCV, validation_data=(X_valCV,-Y_valCV), epochs=100, batch_size=16)
    mapper.save('Model_Summary//Param_Check_Mapper_New//Mapper_Fold_'+ str(j+1) + '.h5')

    #Validation
    estim = mapper.predict(X_valCV, batch_size=16)
    if j >= 7:
        estim_val = mapper.evaluate(X_valCV, Y_valCV, batch_size=16)
        estim_train = mapper.evaluate(X_trainCV, Y_trainCV, batch_size=16)
        print('The train loss is: %.6f' %estim_train)
        print('The validation loss is: %.6f' %estim_val)
        mse = np.mean((Y_valCV - estim)**2)
        print('The mse is: %.6f' %mse)
        NRMSE = np.sqrt(mse/np.mean(Y_valCV**2))
        print('The normalized rmse is: %.6f' %NRMSE)
        loss_fold.append(NRMSE) 
    else:
        estim_val = mapper.evaluate(X_valCV, -Y_valCV, batch_size=16)
        estim_train = mapper.evaluate(X_trainCV, -Y_trainCV, batch_size=16)
        print('The train loss is: %.6f' %estim_train)
        print('The validation loss is: %.6f' %estim_val)
        mse = np.mean((Y_valCV + estim)**2)
        print('The mse is: %.6f' %mse)
        NRMSE = np.sqrt(mse/np.mean(Y_valCV**2))
        print('The normalized rmse is: %.6f' %NRMSE)
        loss_fold.append(NRMSE) 
        
print(loss_fold)  
    

# %%
q = ([0.694048, 0.54285, 0.636227, 0.591225, 0.708878, 0.572256, 0.800799, 0.800799, 0.573618, 0.592885])

test_statistic = np.mean
(a,b,loss_fold_var,c) = jackknife_stats(q, test_statistic, 0.90)
# losses.append([loss_fold_mean, loss_fold_var])
print(loss_fold_var)
# print('Mean NMSE across folds: ' + str(loss_fold_mean) + ' and Variance: ' + str(loss_fold_var))

# %%
j='SYN_16_cube_3_layer'
mapper = CNN_MEP('syn',dr=True,dr_rate=0.15,l_1=0.0001)
mapper.load_weights('Model_Summary//ICASSP_Sims//Weights_'+j+'_.h5')
# mapper.load_weights('Model_Summary//Param_Check//Mapper_Fold_'+ str(7) + '.h5')
mapper.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')

# %%
from numpy import dot, inner
from numpy.linalg import norm

X_check = X_test
Y_check = Y_test
a = range(Y_test.shape[0])
k=16

# mapper = CNN_MEP('res',dr=True,dr_rate=0.15,l_2=0.0001)
# mapper.load_weights('Model_Summary//Param_Check//Mapper_Fold_'+str((2*k)-1)+'.h5')
p1 = mapper.predict(X_check, batch_size=16)
# results = p1
p2 = residual.predict(X_check, batch_size=16)
results = p1 + p2
# mapper.load_weights('Model_Summary//Param_Check//Mapper_Fold_'+str(2*k)+'.h5')
temp_var = 0

cos_sim = np.zeros(Y_check.shape[0])
eps_vec = 1e-1*np.ones(15)

NMSE_syn = np.sqrt(((np.square(Y_check-p1)).mean(axis=None))/((Y_check**2).mean(axis=None)))
for i in a:
    if np.inner(p1[i], Y_check[i]) == 0:
        temp_var = np.inner(p1[i], eps_vec) + np.inner(eps_vec, eps_vec) 
        cos_sim[i] = temp_var / (norm(eps_vec+p1[i])*norm(eps_vec))
    else:
        cos_sim[i] = np.inner(p1[i], Y_check[i])/(norm(p1[i])*(norm(Y_check[i])))
index = np.nonzero(cos_sim)
cosine_syn = np.mean(cos_sim[index])

NMSE_res = np.sqrt(((np.square(Y_check-results)).mean(axis=None))/((Y_check**2).mean(axis=None)))
for i in a:
    if np.inner(results[i], Y_check[i]) == 0:
        temp_var = np.inner(results[i], eps_vec) + np.inner(eps_vec, eps_vec) 
        cos_sim[i] = temp_var / (norm(eps_vec+results[i])*norm(eps_vec))
    else:
        cos_sim[i] = np.inner(results[i], Y_check[i])/(norm(results[i])*(norm(Y_check[i])))
index = np.nonzero(cos_sim)
cosine_res = np.mean(cos_sim[index])
# print(cos_sim[index])

print('The cosine similarity of SYN is: %.4f' %cosine_syn)
print('The cosine similarity of MEP is: %.4f' %cosine_res)
print('The NMSE of SYN is: %.4f' %NMSE_syn)
print('The NMSE of MEP is: %.4f' %NMSE_res)

title_font = {'fontname':'Arial', 'size':'3', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
label_font = {'fontname':'Arial', 'size':'3'}
axis_font = {'fontname':'Arial', 'size':'3'}

title_set = ['Ground Truth', 'SYN' + ', NMSE = %.4f' %NMSE_syn, 'Both' + ', NMSE = %.4f' %NMSE_res]
fig, ax_set = plt.subplots(3,1)

for title, y, ax in zip(title_set, (Y_check[a], p1[a], results), ax_set):
    im = ax.imshow(y.T, cmap='magma', interpolation="none")
    im.set_clim([0.0, 1.0])
    ax.set_ylabel('Muscle index', **label_font)
    ax.set_xlabel('Trial index', **label_font)
    ax.set_title(title, **title_font)
    ax.set_yticks(np.arange(0, Y_test[a].shape[1], 1));
    ax.set_xticks(np.arange(0, Y_test[a].shape[0], 1));
    ax.set_yticklabels(np.arange(1, Y_test[a].shape[1]+1, 1), **axis_font);
    ax.set_xticklabels(np.arange(1, Y_test[a].shape[0]+1, 1), **axis_font)
    ax.set_yticks(np.arange(0.5, Y_test[a].shape[1], 1), minor=False);
    ax.set_xticks(np.arange(0.5, Y_test[a].shape[0], 1), minor=False);
    ax.grid(True, which='minor', color='Silver')
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1], aspect=15)    
    cbar.ax.tick_params(labelsize=4) 
plt.savefig('Model_Summary//ICASSP_Sims_Run_3//Plot_'+j+'_both_.png', transparent=True, dpi=300, format='png')
# plt.savefig('Model_Summary//ICASSP_Sims_Run_2//Plot_'+j+'_SY_.png', transparent=True, dpi=300, format='png')

# %%
j='Both_3_layer_Firing'

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 1.0
    drop = 0.7
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        
    return lrate
callbacks = [LearningRateScheduler(step_decay)]

residual = CNN_MEP('res',final='tanh',activ='linear',dr=True,dr_rate=0.15,l_1=0.0001)
# residual.load_weights('Model_Summary//ICASSP_Sims//Weights_'+j+'_.h5')
# residual.load_weights('Mapper_MEP_Fold_3.h5')
residual.compile(optimizer=Adadelta(lr=1), loss='mean_squared_error')

Y_train_syn = mapper.predict(X_train, batch_size=16)
Y_train_err = Y_train - Y_train_syn

history = residual.fit(X_train, Y_train_err, callbacks=callbacks, epochs=150, batch_size=16)

# with open('Model_Summary//ICASSP_Sims//Summary_'+j+'_.txt', 'w') as f:
#     with redirect_stdout(f):
#         mapper.summary()
# residual.save('Model_Summary//ICASSP_Sims//Weights_'+j+'_.h5')
residual.save('Model_Summary//ICASSP_Sims_Run_3//Weights_'+j+'_v2_.h5')

    #Validation
estim = residual.predict(X_test, batch_size=16)
estim_train = residual.evaluate(X_train, Y_train, batch_size=16)
estim_val = residual.evaluate(X_test, Y_test, batch_size=16)
print('The train loss is: %.6f' %estim_train)
print('The test loss is: %.6f' %estim_val)
mse = np.mean((Y_test - estim)**2)
print('The mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(Y_test**2))
print('The normalized rmse is: %.6f' %NRMSE)    

# %%
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot, inner
from numpy.linalg import norm

X_check = X_test
Y_check = Y_test
a = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

p1 = mapper.predict(X_check, batch_size=16)
p2 = residual.predict(X_check, batch_size=16)
results = p1[a] + p2[a]

# max_val = np.max(results)
# results /= max_val
# max_val = np.max(p1)
# p1 /= max_val

# print('Test:  ' + str(mapper.evaluate(X_test, Y_test, batch_size=4)))
# print('Test:  ' + str(residual.evaluate(X_test, Y_test, batch_size=4)))

cos_sim = np.zeros(Y_check.shape[0])
eps_vec = 1e-1*np.ones(15)

NMSE_syn = (np.square(Y_check - p1)).mean(axis=None)/np.sqrt((Y_check**2).mean(axis=None))
for i in a:
    if np.inner(p1[i], Y_check[i]) == 0:
        temp_var = np.inner(p1[i], eps_vec) + np.inner(eps_vec, eps_vec) 
        cos_sim[i] = temp_var / (norm(eps_vec+p1[i])*norm(eps_vec))
    else:
        cos_sim[i] = np.inner(p1[i], Y_check[i])/(norm(p1[i])*(norm(Y_check[i])))
index = np.nonzero(cos_sim)
cosine_syn = np.mean(cos_sim[index])

NMSE_res = (np.square(Y_check - results)).mean(axis=None)/np.sqrt((Y_check**2).mean(axis=None))
for i in a:
    if np.inner(results[i], Y_check[i]) == 0:
        temp_var = np.inner(results[i], eps_vec) + np.inner(eps_vec, eps_vec) 
        cos_sim[i] = temp_var / (norm(eps_vec+results[i])*norm(eps_vec))
    else:
        cos_sim[i] = np.inner(results[i], Y_check[i])/(norm(results[i])*(norm(Y_check[i])))
index = np.nonzero(cos_sim)
cosine_res = np.mean(cos_sim[index])
print(cos_sim[index])

print('The cosine similarity of SYN is: %.4f' %cosine_syn)
print('The cosine similarity of residual is: %.4f' %cosine_res)
print('The NMSE of SYN is: %.4f' %NMSE_syn)
print('The NMSE of residual is: %.4f' %NMSE_res)

title_font = {'fontname':'Arial', 'size':'5', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
label_font = {'fontname':'Arial', 'size':'4'}
axis_font = {'fontname':'Arial', 'size':'3'}

title_set = ['Ground Truth', 'SYN Only, CS = %.4f' %cosine_syn + ', NMSE = %.4f' %NMSE_syn, 'Residual (SYN->MEP), CS = %.4f' %cosine_res + ', NMSE = %.4f' %NMSE_res]
fig, ax_set = plt.subplots(3,1)

for title, y, ax in zip(title_set, (Y_check[a], p1[a], results), ax_set):
# for title, y, ax in zip(title_set, (Y_check[a], results[a], (Y_train[a]-results[a])), ax_set):
    im = ax.imshow(y.T, cmap='magma', interpolation="none")
    im.set_clim([0.0, 1.0])
    ax.set_ylabel('Muscle index', **label_font)
    ax.set_xlabel('Trial index', **label_font)
    ax.set_title(title, **title_font)
    ax.set_yticks(np.arange(0, Y_test[a].shape[1], 1));
    ax.set_xticks(np.arange(0, Y_test[a].shape[0], 1));
    ax.set_yticklabels(np.arange(1, Y_test[a].shape[1]+1, 1), **axis_font);
    ax.set_xticklabels(np.arange(1, Y_test[a].shape[0]+1, 1), **axis_font)
    ax.set_yticks(np.arange(0.5, Y_test[a].shape[1], 1), minor=False);
    ax.set_xticks(np.arange(0.5, Y_test[a].shape[0], 1), minor=False);
    ax.grid(True, which='minor', color='Silver')
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1], aspect=15)    
    cbar.ax.tick_params(labelsize=4) 
# plt.tight_layout()
# plt.savefig(output_folder + '\\' + output_prefix + '\\' + 'muscle_activation.png', transparent=True, dpi=1500, format='png')
plt.savefig('muscle_activation.png', transparent=True, dpi=600, format='png')


# %%
residual_sig = CNN_MEP('res', 'tanh')
residual_sig.compile(optimizer=Adagrad(lr=0.00001), loss='mean_squared_error', metrics=['accuracy'])
# residual.compile(optimizer=Adadelta(lr=0.005), loss='mean_squared_error', metrics=['accuracy'])
residual_sig.summary()

Y_train_syn = mapper.predict(X_train, batch_size=16)
Y_train_err = Y_train - Y_train_syn

callbacks = []
history = residual_sig.fit(X_train, Y_train_err, callbacks=callbacks, epochs=200, batch_size=16)

# %%
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot, inner
from numpy.linalg import norm

X_check = X_test
Y_check = Y_test
a = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

p1 = mapper.predict(X_check, batch_size=16)
p2 = residual_sig.predict(X_check, batch_size=16)
results = p1[a] + p2[a]

# max_val = np.max(results)
# results /= max_val

# max_val = np.max(p1)
# p1 /= max_val

cos_sim = np.zeros(Y_check.shape[0])
NMSE_syn = (np.square(Y_check - p1)).mean(axis=None)/np.sqrt((Y_check**2).mean(axis=None))
for i in a:
#     cos_sim[i] = np.inner(p1[i], Y_test[i])/(norm(p1[i]+1e-12)*(norm(Y_test[i])+1e-12))
    if np.inner(p1[i], Y_check[i]) == 0:
         cos_sim[i] = 0
    else:
        cos_sim[i] = np.inner(p1[i], Y_check[i])/(norm(p1[i])*(norm(Y_check[i])))
index = np.nonzero(cos_sim)
cosine_syn = np.mean(cos_sim[index])

NMSE_res = (np.square(Y_check - results)).mean(axis=None)/np.sqrt((Y_check**2).mean(axis=None))
# NMSE_res = ((Y_check - results)**2).mean(axis=None)
for i in a:
    if np.inner(results[i], Y_check[i]) == 0:
         cos_sim[i] = 0
    else:
        cos_sim[i] = np.inner(results[i], Y_check[i])/(norm(results[i])*(norm(Y_check[i])))
index = np.nonzero(cos_sim)
cosine_res = np.mean(cos_sim[index])

print('The cosine similarity of synergy is: %.4f' %cosine_syn)
print('The cosine similarity of residual is: %.4f' %cosine_res)
print('The NMSE of synergy is: %.4f' %NMSE_syn)
print('The NMSE of residual is: %.4f' %NMSE_res)

title_font = {'fontname':'Arial', 'size':'5', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} # Bottom vertical alignment for more space
label_font = {'fontname':'Arial', 'size':'4'}
axis_font = {'fontname':'Arial', 'size':'3'}

title_set = ['Ground Truth', 'SYN Only, CS = %.4f' %cosine_syn + ', NMSE = %.4f' %NMSE_syn, 'Residual (SYN->MEP), CS = %.4f' %cosine_res + ', NMSE = %.4f' %NMSE_res]
fig, ax_set = plt.subplots(3,1)

for title, y, ax in zip(title_set, (Y_check[a], p1[a], results), ax_set):
# for title, y, ax in zip(title_set, (Y_check[a], results[a], (Y_train[a]-results[a])), ax_set):
    im = ax.imshow(y.T, cmap='magma', interpolation="none")
    im.set_clim([0.0, 1.0])
    ax.set_ylabel('Muscle index', **label_font)
    ax.set_xlabel('Trial index', **label_font)
    ax.set_title(title, **title_font)
    ax.set_yticks(np.arange(0, Y_test[a].shape[1], 1));
    ax.set_xticks(np.arange(0, Y_test[a].shape[0], 1));
    ax.set_yticklabels(np.arange(1, Y_test[a].shape[1]+1, 1), **axis_font);
    ax.set_xticklabels(np.arange(1, Y_test[a].shape[0]+1, 1), **axis_font)
    ax.set_yticks(np.arange(0.5, Y_test[a].shape[1], 1), minor=False);
    ax.set_xticks(np.arange(0.5, Y_test[a].shape[0], 1), minor=False);
    ax.grid(True, which='minor', color='Silver')
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 0.5, 1], aspect=15)    
    cbar.ax.tick_params(labelsize=7) 
# plt.tight_layout()
# plt.savefig(output_folder + '\\' + output_prefix + '\\' + 'muscle_activation.png', transparent=True, dpi=1500, format='png')
plt.savefig('muscle_activation.png', transparent=True, dpi=600, format='png')

# %%
print(cos_sim[index])

# %%
#Print predictions
fig = plt.figure(figsize=(10,11))

#Original
ax = fig.add_subplot(3,1,1)
plt.pcolor(Y_check[a].T, antialiased=True, cmap='magma')
# plt.pcolor(Y_train[a].T, antialiased=True, cmap='magma')
plt.title('Ground truth')
plt.xlabel('Trial')
plt.ylabel('Muscle Index')

#Predicted
ax = fig.add_subplot(3,1,2)
# plt.pcolor(results[0:19].T, antialiased=True, cmap='magma')
plt.pcolor(results[a].T, antialiased=True, cmap='magma')
plt.title('Prediction')
plt.xlabel('Trial')
plt.ylabel('Muscle Index')

#Difference
# ax = fig.add_subplot(3,1,3)
# # plt.pcolor(results[0:19].T, antialiased=True, cmap='magma')
# plt.pcolor((Y_test[a]-results[a]).T, antialiased=True, cmap='magma')
# plt.title('Difference')
# plt.xlabel('Trial')
# plt.ylabel('Muscle Index')

# plt.show()    
plt.savefig('Prediction.png', transparent=True, dpi=600, format='png')

# %%
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_train, np.zeros(shape=(Y_train.shape[0], 1)))

init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
sess.run(init_op)

for j, (train_idx, val_idx) in enumerate(folds):

    print('\n Running Fold: ', j+1)
    X_train_cv = X_train[train_idx]
    Y_train_cv = Y_train[train_idx]
    X_valid_cv = X_train[val_idx]
    Y_valid_cv = Y_train[val_idx]

    mapper = CNN_MEP()
    # Adagrad(lr=0.0003856944438816557), loss='binary_crossentropy'), mean_absolute_error, mean_squared_error
    mapper.compile(optimizer=Adagrad(lr=0.00002), loss='mean_squared_error', metrics=['accuracy'])
    # 'Adadelta'
    mapper.summary()
    
    callbacks = []
#     callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.75, min_delta = 0.00001, patience=5, verbose=1, min_lr=1e-6, cooldown=10))

    history = mapper.fit(X_train_cv, Y_train_cv,  validation_data=(X_valid_cv, Y_valid_cv), 
                         callbacks=callbacks, epochs=100, batch_size=8)

# %%
#Plot loss
fig, ax = plt.subplots()
import matplotlib.pyplot as plt
# plt.plot(history.history['loss'][0:])
#plt.plot(history.history['binary_crossentropy'])
# plt.title('Model Loss')
#plt.legend(['Training Loss', 'Validation Loss'])
# plt.ylabel('MSE')
# plt.xlabel('epoch')
# plt.show()

a = 28*np.arange( len(history.history['loss'][5:]) )
ax.plot(a, history.history['loss'][5:])
ax.plot(a, history.history['val_loss'][5:])
ax.set_title('Mapper Loss')
ax.set_ylabel('MSE')

ax.set_xlabel('Iterations (dataset / mini-batch)')
ax.legend(['train', 'valid'], loc='upper left')
ax.grid(True)
plt.tight_layout()
plt.savefig('Mapper Loss.png', transparent=True, dpi=600, format='png')

# %%
np.sqrt(np.mean((Y_test-results)**2)/np.mean(Y_test**2))

# %%
h[h==0]

# %%
h = Y_test-results
h[h==0]=float('nan')
fig = plt.figure(figsize=(7,3.5))
plt.hist(h)
plt.title('Prediction')
plt.xlabel('Sample')
plt.ylabel('Muscle')
#plt.xticks([r + 0.5 for r in range(30)], ['1','','','','5','','','','','10','','','','','15','','','','','20','','','','','25','','','','','30'])
#plt.yticks([r + 0.5 for r in range(15)], ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15'])
#plt.savefig(fname='MEP_predictionMEP.png', dpi=200, bbox_inches='tight')
plt.show()

# %%
#Plot results

plt.figure()

r1 = np.arange(len(losses[::4]))
r2 = [x + 0.2 for x in r1]
r3 = [x + 0.4 for x in r1]
r4 = [x + 0.6 for x in r1]

plt.bar(r1, [row[0] for row in losses[::4]], width=0.2, color = 'blue', yerr=np.sqrt([row[1] for row in losses[::4]]), capsize=4)
plt.bar(r2, [row[0] for row in losses[1::4]], width=0.2, color = 'red', yerr=np.sqrt([row[1] for row in losses[1::4]]), capsize=4)
plt.bar(r3, [row[0] for row in losses[2::4]], width=0.2, color = 'green', yerr=np.sqrt([row[1] for row in losses[2::4]]), capsize=4)
plt.bar(r4, [row[0] for row in losses[3::4]], width=0.2, color = 'orange', yerr=np.sqrt([row[1] for row in losses[3::4]]), capsize=4)
plt.xticks([r + 0.3 for r in range(len(losses[::4]))], ['64,32', '32,16', '16,8', '8,4'])

plt.title('Model Comparison')
plt.ylabel('Normalized RMSE')
plt.xlabel('# Filters')
plt.legend(['(3,3,3), (3,3,3)','(3,3,3), (5,5,5)','(5,5,5), (3,3,3)','(5,5,5), (5,5,5)'])
plt.savefig(fname='RMSE_comparison.png', dpi=200, bbox_inches='tight')
plt.show()



