import tensorflow as tf
import tensorflow.keras.layers as tfkl


def if_autoencoder(LM, Pd):
    tf.keras.backend.clear_session()

    input_A = tfkl.Input((256, LM,1)) #DIMENSIONES K x L x C
    input_B = tfkl.Input((256, Pd,1))
    delay = tfkl.Reshape((256, 1, Pd))(input_B)
    
    #encoder
    encoder_0 = tfkl.Conv2D(16, kernel_size=(1,LM), strides=(2,1), name='Input_layer')(input_A)
    encoder_0 = tfkl.BatchNormalization()(encoder_0)

    encoder_1 = tfkl.Conv2D(16, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_1')(encoder_0) 
    encoder_1 = tfkl.BatchNormalization()(encoder_1)

    encoder_2 = tfkl.Conv2D(32, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_2')(encoder_1) 
    encoder_2 = tfkl.BatchNormalization()(encoder_2)

    encoder_3 = tfkl.Conv2D(32, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_3')(encoder_2) 
    encoder_3 = tfkl.BatchNormalization()(encoder_3)

    encoder_4 = tfkl.Conv2D(64, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_4')(encoder_3) 
    encoder_4 = tfkl.BatchNormalization()(encoder_4)

    #latent space
    encoder_5 = tfkl.Conv2D(64, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_5')(encoder_4) 
    encoder_5 = tfkl.BatchNormalization()(encoder_5)

    #decoder
    decoder_6 = tfkl.Conv2DTranspose(64, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_6')(encoder_5)
    decoder_6 = tfkl.BatchNormalization()(decoder_6)
    decoder_6 = tfkl.Add()([encoder_4, decoder_6])
    
    decoder_7 = tfkl.Conv2DTranspose(32, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_7')(decoder_6)
    decoder_7 = tfkl.BatchNormalization()(decoder_7)
    decoder_7 = tfkl.Add()([encoder_3, decoder_7])
    

    decoder_8 = tfkl.Conv2DTranspose(32, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_8')(decoder_7)
    decoder_8 = tfkl.BatchNormalization()(decoder_8)
    decoder_8 = tfkl.Add()([encoder_2, decoder_8])
    
    decoder_9 = tfkl.Conv2DTranspose(16, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_9')(decoder_8)
    decoder_9 = tfkl.BatchNormalization()(decoder_9)
    decoder_9 = tfkl.Add()([encoder_1, decoder_9])
    
    decoder_10 = tfkl.Conv2DTranspose(16, kernel_size=(9,1), strides=(2,1), activation='relu',padding='same',name='hidden_10')(decoder_9)
    decoder_10 = tfkl.BatchNormalization()(decoder_10)
    decoder_10 = tfkl.Add()([encoder_0, decoder_10])
    
    decoder_11 = tfkl.Conv2DTranspose(Pd, kernel_size=(9,1), strides=(2,1), activation='linear',padding='same',name='hidden_11')(decoder_10)

    out = tfkl.multiply([delay, decoder_11])
    out = tf.reduce_sum(out, 3)
    out = tfkl.Activation('relu')(out)

    modelo = tf.keras.Model(inputs=[input_A, input_B], outputs=[out])

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    
    return modelo
