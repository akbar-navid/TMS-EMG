# %%
imports=True
if imports==True:
    import os
    import numpy as np
    from scipy import stats
    import matplotlib as mpl
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    import tensorflow as tf
    from tensorflow.keras import losses
    from tensorflow.keras.layers import Lambda, Dense, Conv3D, MaxPooling3D, BatchNormalization, Dropout, Flatten, Reshape, Conv3DTranspose, UpSampling3D
    from tensorflow.keras.models import Model
    #from keras.utils import to_categorical
    from tensorflow.keras.regularizers import l1_l2
    import tensorflow.keras.backend as K
    K.set_image_data_format('channels_last')
    from tensorflow.keras.activations import relu

    # Set seeds
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
    data_type= np.float32


# %%   
def relu_out(x):
    return tf.minimum(relu(x), 1)


def VAE(main_input,filters=(32,64), ker_size=((3,3,3),(3,3,3)), init='glorot_uniform', bn=False, act='relu',
        act_2='linear',act_f='relu',l_1=0, l_2=0, a_1=0, a_2=0, mu=0, std=1, bias=True):
    
#     main_input = Input(shape=(64,64,64,1), name='input')
    
        #Encoder
    y = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[0], 
               padding='same', activation=act, kernel_initializer=init, name='conv_1')(main_input)
    if bn:
        y = BatchNormalization()(y)
    y  = MaxPooling3D(padding='same', name='pool_1')(y)

    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[1],
               padding='same', activation=act, kernel_initializer=init, name='conv_2')(y)
    if bn:
        y = BatchNormalization()(y)
    y  = MaxPooling3D(padding='same', name='pool_2')(y)    

    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation=act, use_bias=bias, 
                kernel_initializer=init, name='encoded')(y)
        
    flat_enc = Flatten(name='flatten')(y)
    
#     enc_z = Dense(16, activation=act_2,use_bias=bias,name='enc')(flat_enc)
     
    #-----Start of Sampling-----#

    z_mean = Dense(16, activation=act_2, use_bias=bias,name='mean')(flat_enc)
    z_log_var = Dense(16, activation=act_2, use_bias=bias,name='var')(flat_enc) 

    def sampling(args):
        z_mean, z_log_var = args
        shp = K.shape(z_mean)
        epsilon = K.random_normal(shape=shp,
                                    mean=mu, stddev=std)
        return (z_mean + K.exp(0.5 * z_log_var) * epsilon)

    z = Lambda(sampling,name='sampling')([z_mean, z_log_var])

    intermediate_dim = y.get_shape().as_list()
    x = Dense(intermediate_dim[1]*intermediate_dim[2]*intermediate_dim[3]*intermediate_dim[4],
              use_bias=bias,activation=act,name='dense')(z)
    
    y = Reshape(intermediate_dim[1:], name='encoded_alt')(x)
    
    
    #Decoder        
    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[1],
               padding='same', activation=act, kernel_initializer=init, name='conv_up_1')(y)
    if bn:
        y = BatchNormalization()(y)
    y  = UpSampling3D(name='up_1')(y)
    
    y = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[0],
               padding='same', activation=act, kernel_initializer=init, name='conv_up_2')(y)
    if bn:
        y = BatchNormalization()(y)
    y  = UpSampling3D(name='up_2')(y)
    
    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation=act_f, use_bias=bias,
                kernel_initializer=init, name='conv_up_3')(y)
    
    y = Lambda(lambda x: tf.multiply(motor_mask, y), name='final')(y)
        
    final_dim = y.get_shape().as_list()

    def vae_loss(x, x_decoded_mean):
        # NOTE: binary_crossentropy expects a batch_size by dim
        # for x and x_decoded_mean, so we MUST flatten these!
        x = K.flatten(x)
        x_decoded_mean = K.flatten(x_decoded_mean)
#         xent_loss = final_dim[1]*final_dim[2]*final_dim[3] * losses.binary_crossentropy(x, x_decoded_mean)
#         cos_loss = final_dim[1]*final_dim[2]*final_dim[3] * losses.CosineSimilarity(x, x_decoded_mean)
        mse_loss = final_dim[1]*final_dim[2]*final_dim[3] * losses.mean_squared_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return cos_loss + kl_loss
        return mse_loss + kl_loss
    
#     return Model(inputs=main_input, outputs=y)     
    return y, vae_loss


def AE(main_input,filters=(32,64), ker_size=((3,3,3),(3,3,3)), init='glorot_uniform', bn=False, bias=True,
       act='relu',act_f='relu',l_1=0,l_2=0,a_1=0,a_2=0,mu=0,std=1):
    
#     main_input = Input(shape=(64,64,64,1), name='input')
    
        #Encoder
    y = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, 
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[0], 
               padding='same', activation=act, kernel_initializer=init, name='conv_1')(main_input)
    if bn:
        y = BatchNormalization(name='bn_up_1')(y)
    y  = MaxPooling3D(padding='same', name='pool_1')(y)

    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[1],
               padding='same', activation=act, kernel_initializer=init, name='conv_2')(y)
    if bn:
        y = BatchNormalization(name='bn_up_2')(y)
    y  = MaxPooling3D(padding='same', name='pool_2')(y)    

    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation=act, use_bias=bias, 
                kernel_initializer=init, name='encoded')(y)
      
    #Decoder        
    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[1],
               padding='same', activation=act, kernel_initializer=init, name='conv_up_1')(y)
    if bn:
        y = BatchNormalization(name='bn_down_1')(y)
    y  = UpSampling3D(name='up_1')(y)
    
    y = Conv3D(filters=filters[0], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=ker_size[0],
               padding='same', activation=act, kernel_initializer=init, name='conv_up_2')(y)
    if bn:
        y = BatchNormalization(name='bn_down_2')(y)
    y  = UpSampling3D(name='up_2')(y)
    
    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation=act_f, use_bias=bias,
                kernel_initializer=init, name='conv_up_3')(y)
    
    y = Lambda(lambda x: tf.multiply(motor_mask, y))(y)
    
    return y


def Inv_Map(inputs, act_f='relu', dr=False, dr_rate=0, bn=False,l_1=0, l_2=0, act='relu',
            filters=(16,32,64), init='glorot_uniform', bn_d=False, bias=True):
    
    # Phase 1
    x = Dense(units=32, activation=act, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, name='fc_1')(inputs)
    if dr:
        x = Dropout(dr_rate)(x) 
 
    x = Dense(units=128, activation=act, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, name='fc_2')(x)
    if dr:
        x = Dropout(dr_rate)(x) 
        
    x = Dense(units=512, activation=act, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, name='fc_3')(x)
    if dr:
        x = Dropout(dr_rate)(x) 
        
    x = Reshape((2, 2, 2,64))(x)
    
    # Phase 2    
    x  = UpSampling3D(name='up_1a')(x)
    x = Conv3DTranspose(filters=filters[2], kernel_size=(3,3,3), padding='same',
                        activation=act, use_bias=bias, 
                        kernel_regularizer=l1_l2(l1=l_1, l2=l_2), name='conv_1a')(x)    
    if bn:
        x  = BatchNormalization()(x)  
    
    x  = UpSampling3D(name='up_2a')(x) 
    x = Conv3DTranspose(filters=filters[1], kernel_size=(3,3,3), padding='same',
                        activation=act, use_bias=bias, 
                        kernel_regularizer=l1_l2(l1=l_1, l2=l_2), name='conv_2a')(x)   
    if bn:
        x  = BatchNormalization()(x)        
 
    x  = UpSampling3D(name='up_3a')(x) 
    x = Conv3DTranspose(filters=1, kernel_size=(3,3,3), padding='same', use_bias=bias, 
                        kernel_initializer=init, activation=act, name='conv_f')(x)    
    
    #Decoder        
    y = Conv3D(filters=filters[2], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               kernel_size=(3,3,3), padding='same', activation=act, kernel_initializer=init, name='conv_up_1')(x)
    y  = UpSampling3D(name='up_1')(y)
    if bn:
        y = BatchNormalization(name='bn_down_1')(y)
    
    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, kernel_size=(3,3,3),
               padding='same', activation=act, kernel_initializer=init, name='conv_up_2')(y)
    y  = UpSampling3D(name='up_2')(y)
    if bn:
        y = BatchNormalization(name='bn_down_2')(y)
    
    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation=act_f, use_bias=bias,
                kernel_initializer=init, name='conv_up_3')(y)
    
    y = Lambda(lambda x: tf.multiply(motor_mask, y))(y)     
    
    return Model(inputs=inputs, outputs=y)


def Inv_Map_VAE_Lat_2d(inputs, motor_mask, l_1=0, l_2=0, act='relu', act_2='linear', act_3='relu', bias=True, vae_std=False, vae_dec=False,act_f='relu', filters=(16,32,64), init='glorot_uniform', mu=0, std=1, a_1=0,a_2=0):
    
    enc_z = Dense(units=16, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, name='fc_1')(inputs)
    enc_z = Dense(units=32, kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias, name='fc_2')(enc_z)

    def relu_out(x):
        return tf.minimum(relu(x), 1)
    
    if vae_dec:
        z = Dense(units=16, use_bias=bias, name='fc_3')(enc_z) 
#         z = Dense(units=16, use_bias=bias, name='sampling_2')(enc_z)    # Only for Sub-1 (old)
    
    elif vae_std:
        enc_z = Dense(units=4096, use_bias=bias, name='fc_3')(enc_z)    
        z_mean = Dense(16, use_bias=bias,name='mean')(enc_z)
        z_log_var = Dense(16, use_bias=bias,name='var')(enc_z)     

        def sampling(args):
            z_mean, z_log_var = args
            shp = K.shape(z_mean)
            epsilon = K.random_normal(shape=shp,
                                        mean=mu, stddev=std)
            return (z_mean + K.exp(0.5 * z_log_var) * epsilon)

        z = Lambda(sampling,name='sampling')([z_mean, z_log_var])
        
    else:
        z_mean = Dense(16, use_bias=bias,name='mean_2')(enc_z)
        z_log_var = Dense(16, use_bias=bias,name='var_2')(enc_z)     

        def sampling(args):
            z_mean, z_log_var = args
            shp = K.shape(z_mean)
            epsilon = K.random_normal(shape=shp,
                                        mean=mu, stddev=std)
            return (z_mean + K.exp(0.5 * z_log_var) * epsilon)

        z = Lambda(sampling,name='sampling')([z_mean, z_log_var])
#         z = Lambda(sampling,name='sampling_2')([z_mean, z_log_var])    # Only for Sub-1? (old)

    x = Dense(4096, use_bias=bias,activation=act_3,name='dense')(z)
    
    y = Reshape((16, 16, 16,1), name='encoded_alt')(x)
    
    
    #Decoder        
    y = Conv3D(filters=filters[2], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=(3,3,3),
               padding='same', activation=act, kernel_initializer=init, name='conv_up_1')(y)
    y  = UpSampling3D(name='up_1')(y)
    
    y = Conv3D(filters=filters[1], kernel_regularizer=l1_l2(l1=l_1, l2=l_2), use_bias=bias,
               activity_regularizer=l1_l2(l1=a_1, l2=a_2), kernel_size=(3,3,3),
               padding='same', activation=act, kernel_initializer=init, name='conv_up_2')(y)
    y  = UpSampling3D(name='up_2')(y)
    
    y  = Conv3D(filters=1, kernel_size=(3,3,3), padding='same', activation=act_f, use_bias=bias,
                kernel_initializer=init, name='conv_up_3')(y)
    
    y = Lambda(lambda x: tf.multiply(motor_mask, y))(y)    
            
    if vae_dec:
        return y     
    
    else:
        final_dim = y.get_shape().as_list()

        def loss_map(x, x_decoded_mean):
            # NOTE: x and x_decoded_mean must be flattened
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            mse_loss = final_dim[1]*final_dim[2]*final_dim[3] * losses.mean_squared_error(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return mse_loss + kl_loss

        return y, loss_map  


def plot_assist(Y_test_CV,NRMSE_all,Y):
    '''Calculate necessary values for box-plot with NRMSE'''
    
    MEP_av = []
    MEP_active_av = []
    MEP_var = []
    MEP_active_var = []
    Muscles_Active = [] 

    for q in range(len(Y_test_CV)):        
        MEP_av.append(np.mean(Y_test_CV[q])) 
        Muscles_Active.append(np.count_nonzero(Y_test_CV[q]))
        active_mean = np.sum(Y_test_CV[q])/np.count_nonzero(Y_test_CV[q])
        MEP_active_av.append(active_mean)
        MEP_var.append(np.var(Y_test_CV[q])) 
        MEP_active_var.append(np.sum(Y_test_CV[q]**2)/np.count_nonzero(Y_test_CV[q]) - active_mean**2)
    
    return MEP_av, MEP_active_av, MEP_var, MEP_active_var, Muscles_Active


def plot_box(Y_index,Y_train,rmse_all,nrmse_all):
    '''Calculate necessary values for box-plot with NRMSE'''
    
    for x in range(16):
        globals()['a%s' % x] = []
        globals()['b%s' % x] = []
    
    j = 0
    for q in Y_index: 
        if np.count_nonzero(Y_train[q]) == 1:
            a1.append(rmse_all[j])
            b1.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 2:
            a2.append(rmse_all[j])
            b2.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 3:
            a3.append(rmse_all[j])
            b3.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 4:
            a4.append(rmse_all[j])
            b4.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 5:
            a5.append(rmse_all[j])
            b5.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 6:
            a6.append(rmse_all[j])
            b6.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 7:
            a7.append(rmse_all[j])
            b7.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 8:
            a8.append(rmse_all[j])
            b8.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 9:
            a9.append(rmse_all[j])
            b9.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 10:
            a10.append(rmse_all[j])
            b10.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 11:
            a11.append(rmse_all[j])
            b11.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 12:
            a12.append(rmse_all[j])
            b12.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 13:
            a13.append(rmse_all[j])
            b13.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 14:
            a14.append(rmse_all[j])
            b14.append(nrmse_all[j])
        elif np.count_nonzero(Y_train[q]) == 15:
            a15.append(rmse_all[j])
            b15.append(nrmse_all[j])
        j += 1

    box_rmse = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15]
    box_nrmse = [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15]

    return box_rmse, box_nrmse


def calc_nrmse_r_sq(X_nrmse,X_test_CV):
    '''Calculate NRMSE and R_square for a given E-field recon'''      
    
    mse = np.mean((X_test_CV - X_nrmse)**2)
    RMSE = np.sqrt(mse)
    img_pow = np.mean(X_test_CV**2)
    NRMSE = np.sqrt(mse/img_pow)
    
    a = X_nrmse.reshape(262144,1)
    b = X_test_CV.reshape(262144,1)
    
    x = np.zeros(len(np.nonzero(b)[0]), dtype = data_type) 
    y = np.zeros(len(np.nonzero(b)[0]), dtype = data_type)
    a_ind = 0
    b_ind = 0

    for l in range(262144):   
        if a[l] > 0.:  
            x[a_ind] = a[l]
            a_ind += 1

    for l in range(262144):   
        if b[l] > 0.:     
            y[b_ind] = b[l]
            b_ind += 1       

    corr,_ = stats.pearsonr(x,y)
    R_sq = corr**2
    
    return NRMSE, R_sq, RMSE


def find_RMT(Y_test_CV,Y,p1,p2,p3):
    
    """Find RMT level corresponding to passed stim"""
    stim_location = np.argwhere(np.all((Y-Y_test_CV)==0, axis=1))
    if stim_location.shape[0]>1:
        stim_location = stim_location[0,:]
        
    if stim_location<=p1:
        RMT_level = 110
    elif stim_location>p1 and stim_location<=p2:
        RMT_level = 120
    elif stim_location>p2 and stim_location<=p3:
        RMT_level = 130
    else:
        RMT_level = 140
        
    return RMT_level


def RMT_level_stats(RMT_level,NRMSE_all,R_sq_all):    
#     count_1 = 0
    for i in np.unique(RMT_level):
        ind = np.argwhere(RMT_level==i)
        a, b = np.mean(NRMSE_all[ind]), 1.96*stats.sem(NRMSE_all[ind],ddof=0)
        c, d = np.mean(R_sq_all[ind]), 1.96*stats.sem(R_sq_all[ind],ddof=0)   
#         a, b = np.mean(NRMSE_all[count_1:count_1+count_2]), 1.96*stats.sem(NRMSE_all[count_1:count_1+count_2],ddof=0)
#         c, d = np.mean(R_sq_all[count_1:count_1+count_2]), 1.96*stats.sem(R_sq_all[count_1:count_1+count_2],ddof=0)        
#         count_1 += count_2
        if i==110:
            r_110=[a,b,c,d]
        elif i==120:
            r_120=[a,b,c,d]
        elif i==130:
            r_130=[a,b,c,d]
        elif i==140:
            r_140=[a,b,c,d]
            
    return r_110,r_120,r_130,r_140    

