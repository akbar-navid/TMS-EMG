# %%
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from itertools import product
from scipy import io
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Conv3D, UpSampling3D, MaxPooling3D, concatenate, Lambda, BatchNormalization
from keras.models import Model, load_model
#from keras.utils import to_categorical
from keras.initializers import glorot_uniform, he_uniform, he_normal
from keras.optimizers import Adam, SGD, Adadelta
from keras.callbacks import LearningRateScheduler
from keras.utils import normalize
#from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from mpl_toolkits.mplot3d import Axes3D
from keras.regularizers import l1, l2, l1_l2

%matplotlib inline

# %%
# OLDER
np.random.seed(17)
n_files          = 699
ds = 1
data_folder      = '2_Masked_Firing_Norm_Total'
muscle_data_name = 'MEP_All_RMT_Stims'
muscle_var_name  = 'MEP_All_RMT_Stims'
img_shape        = (64, 64, 64)
data_type        = np.float32
data = np.zeros((n_files, *img_shape, 1), dtype = data_type) 
# a = np.zeros((4,n_files, *img_shape, 1), dtype = data_type) 

# Load MEP data
tmp_mat          = io.loadmat('{0}.mat'.format(muscle_data_name))
Y                = np.array(tmp_mat[muscle_var_name][range(n_files)], dtype=data_type)

# Load image data
for i in range(300):   
    tmp_mat = io.loadmat('{0}//RMT110_{1}.mat'.format(data_folder, i+1)) 
    data[i,:,:,:,0] = np.array(tmp_mat['zone'], dtype=data_type)[::ds,::ds,::ds] 

for i in range(149):   
    tmp_mat = io.loadmat('{0}//RMT120_{1}.mat'.format(data_folder, i+1)) 
    data[300+i,:,:,:,0] = np.array(tmp_mat['zone'], dtype=data_type)[::ds,::ds,::ds] 

for i in range(150):   
    tmp_mat = io.loadmat('{0}//RMT130_{1}.mat'.format(data_folder, i+1)) 
    data[449+i,:,:,:,0] = np.array(tmp_mat['zone'], dtype=data_type)[::ds,::ds,::ds] 

for i in range(100):   
    tmp_mat = io.loadmat('{0}//RMT140_{1}.mat'.format(data_folder, i+1)) 
    data[599+i,:,:,:,0] = np.array(tmp_mat['zone'], dtype=data_type)[::ds,::ds,::ds] 

tmp_mat = io.loadmat('M1_mask.mat')
motor_mask = tmp_mat['mask'][::ds,::ds,::ds] # Samples every alternate value, i.e. by 'ds'.
motor_mask = motor_mask.reshape(1,*img_shape,1)

motor_mask = tf.convert_to_tensor(motor_mask, np.float32)

# Split Data
a1, b1, c1, d1 = train_test_split(data[:300,:], Y[:300,:], test_size=0.1, random_state=42)
a2, b2, c2, d2 = train_test_split(data[300:449,:], Y[300:449,:], test_size=0.1, random_state=42)
a3, b3, c3, d3 = train_test_split(data[449:599,:], Y[449:599,:], test_size=0.1, random_state=42)
a4, b4, c4, d4 = train_test_split(data[599:,:], Y[599:,:], test_size=0.1, random_state=42)

# Load Data from All RMTs
X_train = a1
X_train = np.append(X_train,a2, axis=0)
X_train = np.append(X_train,a3, axis=0)
X_train = np.append(X_train,a4, axis=0)

X_test = b1
X_test = np.append(X_test,b2, axis=0)
X_test = np.append(X_test,b3, axis=0)
X_test = np.append(X_test,b4, axis=0)


Y_train = c1
Y_train = np.append(Y_train,c2, axis=0)
Y_train = np.append(Y_train,c3, axis=0)
Y_train = np.append(Y_train,c4, axis=0)

Y_test = d1
Y_test = np.append(Y_test,d2, axis=0)
Y_test = np.append(Y_test,d3, axis=0)
Y_test = np.append(Y_test,d4, axis=0)

np.random.seed(17)
q1 = list(range(Y_train.shape[0]))
q2 = list(range(Y_test.shape[0]))
np.random.shuffle(q1)
np.random.shuffle(q2)

# Reshuffled Data
X_train = X_train[q1]
Y_train = Y_train[q1]
X_test  = X_test[q2]
Y_test  = Y_test[q2]

# Save Shuffled Data
np.save('2_Masked_Firing_Norm_Python_Run_3//Y_All_RMT_Train', Y_train)
np.save('2_Masked_Firing_Norm_Python_Run_3//Y_All_RMT_Test', Y_test)
np.save('2_Masked_Firing_Norm_Python_Run_3//X_All_RMT_Train', X_train)
np.save('2_Masked_Firing_Norm_Python_Run_3//X_All_RMT_Test', X_test)

# print ("X_train shape: " + str(X_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Mask shape: " + str(motor_mask.shape))
# print ("data shape[0]: " + str(data.shape[0]))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("Y_test shape: " + str(Y_test.shape))

print ("X_train shape: " + str(a1.shape))
print ("X_test shape: " + str(b1.shape))
print ("Mask shape: " + str(c1.shape))
print ("data shape[0]: " + str(d1.shape))
print ("Y_train shape: " + str(a2.shape))
print ("Y_test shape: " + str(b2.shape))
print ("Y_train shape: " + str(c2.shape))
print ("Y_test shape: " + str(d2.shape))
print(q1)
print(q2)

# %%
# File Locations and Load Data
n_files          = 699
ds = 1
muscle_data_name = 'MEP_All_RMT_Stims'
muscle_var_name  = 'MEP_All_RMT_Stims'
img_shape        = (64, 64, 64)
data_type        = np.float32

# Load Mask
tmp_mat = io.loadmat('M1_mask.mat')
motor_mask = tmp_mat['mask'][::ds,::ds,::ds] # Samples every alternate value, i.e. by 'ds'.
motor_mask = motor_mask.reshape(1,*img_shape,1)
motor_mask = tf.convert_to_tensor(motor_mask, np.float32)

# X & Y Data
X_train = np.load('2_Masked_Firing_Norm_Total_Python//X_All_RMT_Train.npy').reshape(629, 64, 64, 64, 1)
Y_train = np.load('2_Masked_Firing_Norm_Total_Python//Y_All_RMT_Train.npy').reshape(629, 15)
X_test  = np.load('2_Masked_Firing_Norm_Total_Python//X_All_RMT_Test.npy').reshape(70, 64, 64, 64, 1)
Y_test  = np.load('2_Masked_Firing_Norm_Total_Python//Y_All_RMT_Test.npy').reshape(70, 15)

print ("X_train shape: " + str(X_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Mask shape: " + str(motor_mask.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_test shape: " + str(Y_test.shape))

# %%
#Print several E-fields
num = 4 #Number of pictures to show
fig = plt.figure(figsize=(12,4))

for i in range(num):
    ax = fig.add_subplot(1,5,i+1, projection='3d')
    ax.view_init(90, -90)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)
    idx = np.random.randint(1,X_train.shape[0])
#     idx = i+280
    ax.plot_surface(x,y,np.sum(X_train[idx].reshape(*img_shape),2), cmap=plt.cm.jet, antialiased=False);
    plt.title(idx)
    plt.axis('off');

# %%
def AE_CNN(input_shape=(64,64,64,1), filters=(32,64), ker_size=((3,3,3),(3,3,3)), init='glorot_uniform',
          l_1=0, l_2=0, a_1=0, a_2=0):
    
    inputs = Input(shape=input_shape, name='input')
    bn = False
#     bn = True
    
    #Encoder
    y = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[0], padding='same', activation='elu', kernel_initializer=init, name='conv1_5')(inputs)
    if bn:
        y = BatchNormalization()(y)
    y  = MaxPooling3D(padding='same', name='pool_1')(y)
    
    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[1], padding='same', activation='elu', kernel_initializer=init, name='conv2_3')(y)
    if bn:
        y = BatchNormalization()(y)
    y  = MaxPooling3D(padding='same', name='pool_2')(y)    
    
    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='elu', kernel_initializer=init, name='encoded_alt')(y)
    if bn:
        y = BatchNormalization(name='encoded')(y)
    
    
    #Decoder        
    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[1], padding='same', activation='elu', kernel_initializer=init, name='conv7_5')(y)
    if bn:
        y = BatchNormalization()(y)
    y  = UpSampling3D(name='up_7')(y)
    
    y = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[0], padding='same', activation='elu', kernel_initializer=init, name='conv8_5')(y)
    if bn:
        y = BatchNormalization()(y)
    y  = UpSampling3D(name='up_8')(y)
    
    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation='linear', kernel_initializer=init, name='conv9_3')(y)
    if bn:
        y = BatchNormalization()(y)
    
    y = Lambda(lambda x: tf.multiply(motor_mask, y))(y)
    
    autoencoder = Model(inputs=inputs, outputs=y)
    
    return autoencoder

# %%
#Autoencoder model
autoencoder = AE_CNN(l_1=1e-4)
# autoencoder.summary()
from contextlib import redirect_stdout

#Encoder model
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_alt').output)
# encoder.summary()

# with open('Model_Summary//ICASSP_Sims//Summary_AE_16_Cube_Firing_run_2.txt', 'w') as f:
#     with redirect_stdout(f):
#         autoencoder.summary()

# %%
encoder.load_weights('Encoder_All_RMT.h5')
w1, b1 = encoder.layers[1].get_weights()
print('Max weight is: ', str(np.amax(w1)))
print('Min weight is: ', str(np.amin(w1)))
print('Shape of weight matrix is: ', str(w1.shape))
# print(w1)

# %%
init_op = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
sess.run(init_op)

# %%
#Compilation and train

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 1
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    if epoch>3000:
        lrate=0.1/2
        
    return lrate

callbacks = [LearningRateScheduler(step_decay)]

# autoencoder.load_weights('encoder.h5')
autoencoder.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='mse')
# autoencoder.compile(optimizer=Adadelta(lr=.1), loss='mse')
history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=4, callbacks=callbacks, verbose=1)

encoder.save('2_Masked_Firing_Norm_Python_Run_3//Encoder_Firing_run_3.h5')

# Validation
estim = autoencoder.predict(X_test, batch_size=4)
estim_val = autoencoder.evaluate(X_test, X_test, batch_size=4)
estim_train = autoencoder.evaluate(X_train, X_train, batch_size=4)
print('The train loss is: %.6f' %estim_train)
print('The test loss is: %.6f' %estim_val)

mse = np.mean((X_test - estim)**2)
print('The mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(X_test**2))
print('The normalized rmse is: %.6f' %NRMSE)

# %%
encoder.save('2_Masked_Firing_Norm_Python_Run_3//Encoder_Firing_run_3.h5')

# Validation
estim = autoencoder.predict(X_test, batch_size=4)
estim_val = autoencoder.evaluate(X_test, X_test, batch_size=4)
estim_train = autoencoder.evaluate(X_train, X_train, batch_size=4)
print('The train loss is: %.6f' %estim_train)
print('The test loss is: %.6f' %estim_val)

mse = np.mean((X_test - estim)**2)
print('The mse is: %.6f' %mse)
NRMSE = np.sqrt(mse/np.mean(X_test**2))
print('The normalized rmse is: %.6f' %NRMSE)

# %%

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from keras.regularizers import l1, l2, l1_l2
folds = StratifiedKFold(n_splits=8, shuffle=True, random_state=0).split(X_train, np.zeros(shape=(X_train.shape[0], 1)))
from contextlib import redirect_stdout
        
for j, (train_idx, val_idx) in enumerate(folds):

    # Train and validation split
    print('\n Running Fold: ', j+1)
    X_trainCV = X_train[train_idx]
    X_valCV = X_train[val_idx]  
    loss_fold = [] #List of the K different losses for every fold
        
        #Model 
    if j == 0:
        autoencoder = AE_CNN()
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mae')
    elif j == 1:
        autoencoder = AE_CNN()
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
    elif j == 2:
        autoencoder = AE_CNN(l_1=1e-6)
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
    elif j == 3:
        autoencoder = AE_CNN(l_1=1e-5)
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
    if j == 4:
        autoencoder = AE_CNN(l_1=1e-4)
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
    elif j == 5:
        autoencoder = AE_CNN(l_1=1e-3)
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
    elif j == 6:
        autoencoder = AE_CNN(l_1=1e-2)
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
#     elif j == 7:
#         autoencoder = AE_CNN(a_1=1e-5)
#         autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mae')
#     elif j == 8:
#         autoencoder = AE_CNN(a_1=1e-4)
#         autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mae')
    else:
        autoencoder = AE_CNN(l_1=1e-1)
        autoencoder.compile(optimizer=Adadelta(lr=0.1), loss='mse')
        
#   Autoencoder Summary
    with open('Model_Summary//AE_Final//Summary_Fold_'+str(j)+'_.txt', 'w') as f:
        with redirect_stdout(f):
            autoencoder.summary()

    #Train data
        autoencoder.fit(X_trainCV, X_trainCV, validation_data=(X_valCV,X_valCV), epochs=20, batch_size=4)
#         mapper.save('Model_Summary//Param_Check//Mapper_Fold_'+ str(j+1) + '.h5')

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded_alt').output)
    encoder.save('Model_Summary//AE_Final//Encoder_Fold_'+ str(j+1) + '.h5')
    #Validation
    estim = autoencoder.predict(X_test, batch_size=4)
    estim_val = autoencoder.evaluate(X_valCV, X_valCV, batch_size=4)
    estim_train = autoencoder.evaluate(X_trainCV, X_trainCV, batch_size=4)
    print('The train loss is: %.6f' %estim_train)
    print('The validation loss is: %.6f' %estim_val)
    
    mse = np.mean((X_test - estim)**2)
    print('The mse is: %.6f' %mse)
    NRMSE = np.sqrt(mse/np.mean(X_test**2))
    print('The normalized rmse is: %.6f' %NRMSE)
    loss_fold.append(NRMSE) 
print(loss_fold)  

# %%
#Plot loss
plt.plot(history.history['loss'][5:])
#plt.plot(history.history['binary_crossentropy'])
plt.title('Model Loss')
#plt.legend(['Training Loss', 'Validation Loss'])
plt.ylabel('MSE')
plt.xlabel('epoch')
plt.show()

# %%
#Predictions in test set
encoded_imgs = encoder.predict(X_test, batch_size=4)
decoded_imgs = autoencoder.predict(X_test, batch_size=4)
print('Train set performance: ')
print(autoencoder.evaluate(X_train, X_train, batch_size=4))
print('Test set performance: ')
print(autoencoder.evaluate(X_test, X_test, batch_size=4))

# io.savemat('original', mdict={'original': X_test[27].reshape(64,64,64)})
# io.savemat('encoded', mdict={'encoded': encoded_imgs[27].reshape(16,16,16)})
# io.savemat('decoded', mdict={'decoded': decoded_imgs[27].reshape(64,64,64)})

# %%
encoder.load_weights('Model_Summary//ICASSP_Sims//Encoder_Firing_run_2.h5')
encoded_imgs = encoder.predict(X_test, batch_size=4)

# %%
io.savemat('matlab files//Viz//encoded_test.mat', mdict={'encoded_test': encoded_imgs[27].reshape(16,16,16)})

# %%
print(autoencoder.metrics_names)

# %%
#Print several E-fields
# np.save('Model_Summary//ICASSP_Sims//training_loss_.npy', history.history['loss'])
num = 1 #Number of pictures to show
fig = plt.figure(figsize=(12,30))

for i in range(num):
    idx = np.random.randint(1,X_test.shape[0])
    
    #Original image
    ax = fig.add_subplot(3,num,i+1, projection='3d')
#     ax.view_init(90, -90)
    ax.view_init(90, 180)
#     ax.view_init(85,40)
    x = np.linspace(1,64,64)
    y = np.linspace(1,64,64)
    x, y = np.meshgrid(x,y)
    ax.plot_surface(x,y,np.sum(X_test[idx].reshape(64,64,64),2), cmap=plt.cm.jet, antialiased=False);
    plt.title(idx)
    plt.axis('off');
    
    #Encoded image
    ax = fig.add_subplot(3,num,i+1+num, projection='3d')
    ax.view_init(90, 180)
#     ax.view_init(90, -90)
#     ax.view_init(85,40)
    x = np.linspace(1,16,16)
    y = np.linspace(1,16,16)
    x, y = np.meshgrid(x,y)
    ax.plot_surface(x,y,np.sum(encoded_imgs[idx].reshape(16,16,16),2), cmap=plt.cm.jet, antialiased=False);
    plt.axis('off');
    
#     #Decoded image
#     ax = fig.add_subplot(3,num,i+1+num*2, projection='3d')
#     ax.view_init(90, -90)
#     x = np.linspace(1,64,64)
#     y = np.linspace(1,64,64)
#     x, y = np.meshgrid(x,y)
#     ax.plot_surface(x,y,np.sum(decoded_imgs[idx].reshape(64,64,64),2), cmap=plt.cm.jet, antialiased=False);
#     plt.axis('off');

# %%
#Save Encoded Images

encoded_train = encoder.predict(X_train, batch_size=4)
encoded_test  = encoder.predict(X_test, batch_size=4)
np.save('2_Masked_Firing_Norm_Python_Run_3//Encoded_All_RMT_Firing_v3_Train', encoded_train)
np.save('2_Masked_Firing_Norm_Python_Run_3//Encoded_All_RMT_Firing_v3_Test', encoded_test)


