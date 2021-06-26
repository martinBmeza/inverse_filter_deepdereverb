from model.net import if_autoencoder
from model.data_loader import build_generators
import math
import tensorflow as tf

params = {'path':'/home/martin/tesis/inverse_filter_deepdereverb/data/npy_data/', 
        'batch_size' : 32, 
        'dim' : (256, 256)}
training_generator, validation_generator = build_generators(params)

initial_learning_rate = 0.001
def lr_step_decay(epoch, lr):
    drop_rate = 0.9
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))


#cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=2),
#        tf.keras.callbacks.ModelCheckpoint('/home/martin/Documents/tesis/src/model/ckpts/weights.{epoch:02d}-{loss:.3f}.hdf5'),
#        tf.keras.callbacks.TensorBoard(log_dir='tb_logs',profile_batch=0, update_freq='batch', histogram_freq=1)]

modelo = if_autoencoder(LM=5, Pd=9)
history = modelo.fit(training_generator, epochs=200, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_step_decay, verbose=1)])
modelo.save_weights('/home/martin/tesis/inverse_filter_deepdereverb/model/ckpts/weights.hdf5')
