import sys
sys.path.append('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese')


import os
import pandas as pd
from architectures.similarity_detection import pretext_task_one_nopool, pretext_one, pretext_task_one_aspp
from utils.log_params import log_params
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import zoom
import time
import uuid
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def feature_scaling(img, method):
    I = img
    if method == "STAND":
        I = (I - I.mean()) / I.std()
        return I
    if method == "MINMAX":
        I = ((I - np.nanmin(I))/(np.nanmax(I) - np.nanmin(I)))
        return I
    if method == "MEAN":
        I = (I - I.mean()) / (np.nanmax(I) - np.nanmin(I))
    else:
        return I
    
# ένα callback για να σταματήσει η εκπαίδευση όταν δούμε 90% accuracy
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_loss')<0.080):
            print("\nReached 0.080 validation loss so cancelling training!")
            self.model.stop_training = True

def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""
    # crop if necesary
    I = I[:s[0],:s[1]]
    si = I.shape

    # pad if necessary 
    p0 = max(0,s[0] - si[0])
    p1 = max(0,s[1] - si[1])

    return np.pad(I,((0,p0),(0,p1)),'edge')



def generate_short_id():
    # Generate a UUID
    unique_id = uuid.uuid4()

    # Convert UUID to a hex string and take the first 4 characters
    short_id = str(unique_id.hex)[:4]

    return short_id

s2mtcp_target = '/data/aleoikon_data/change_detection/ssl/s2mtcp/patches/task1/'

#s2mtcp_target = '/home/aleoikon/Documents/data/ssl/s2mtcp/patches_colorshifted/task1/'

df = pd.read_csv(s2mtcp_target+'dataset_unclouded.csv', dtype=str)
train = df.sample(frac=0.85,random_state=1)
validation = df.drop(train.index)
test = validation.sample(frac = 0.33, random_state=1)
validation = validation.drop(test.index)

print("Data", len(df))
print("85% of Data = Train", len(train))
print("10% of Data = Validation", len(validation))
print("5% of Data = Test", len(test))

test_balance = validation['overlap']
(unique, counts) = np.unique(test_balance , return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

n_ch = 3
channel = 'rgb'
method = 'STAND'

X_train1 = np.ndarray(shape=(len(train),96,96,n_ch))
X_train2 = np.ndarray(shape=(len(train),96,96,n_ch))
y_train = np.ndarray(shape=(len(train),1))
X_val1 = np.ndarray(shape=(len(validation),96,96,n_ch))
X_val2 = np.ndarray(shape=(len(validation),96,96,n_ch))
y_val = np.ndarray(shape=(len(validation),1))

def create_rgb(x,channel):
    if channel == 'red':
        r = x[:,:,1]
        r = np.expand_dims(r, axis=2)
        return r
    if channel == 'green':
        g = x[:,:,2]
        g = np.expand_dims(g, axis=2)
        return g
    if channel == 'blue':
        b  = x[:,:,3]
        b = np.expand_dims(b, axis=2)
        return b
    if channel == 'rgb':
        r = x[:,:,1]
        g = x[:,:,2]
        b  = x[:,:,3]
        rgb = np.dstack((r,g,b))
        return(rgb)
    if channel == 'rgbvnir':
        r = x[:,:,1]
        g = x[:,:,2]
        b  = x[:,:,3]
        vnir = x[:,:,8]
        rgbvnir = np.stack((r,g,b,vnir),axis=2).astype('float')
        #rgb = np.dstack((r,g,b))
        return(rgbvnir)
    if channel == 'eq20':
        r = x[:,:,1]
        s = r.shape
        ir1 = adjust_shape(zoom(x[:,:,4],2),s)
        ir2 = adjust_shape(zoom(x[:,:,5],2),s)
        ir3 = adjust_shape(zoom(x[:,:,6],2),s)
        nir2 = adjust_shape(zoom(x[:,:,8],2),s)
        swir2 = adjust_shape(zoom(x[:,:,11],2),s)
        swir3 = adjust_shape(zoom(x[:,:,12],2),s)
        x = np.stack((ir1,ir2,ir3,nir2,swir2,swir3),axis=2).astype('float') 
        return x
    else:
        return x
        print("NOT CORRECT CHANNELS")
pos = 0
for index in train.index:
    img1 = np.load(s2mtcp_target + train['pair1'][index])
    img2 = np.load(s2mtcp_target + train['pair2'][index])
    X1 = create_rgb(img1,channel)
    X2 = create_rgb(img2, channel)
    X1 = feature_scaling(X1, method)
    X2 = feature_scaling(X2, method)
    X_train1[pos] = X1
    X_train2[pos] = X2
    y_train[pos] = train['overlap'][index]
    pos += 1

pos = 0
for index in validation.index:
    img1 = np.load(s2mtcp_target + validation['pair1'][index])
    img2 = np.load(s2mtcp_target + validation['pair2'][index])
    X1 = create_rgb(img1,channel)
    X2 = create_rgb(img2, channel)
    X1 = feature_scaling(X1, method)
    X2 = feature_scaling(X2, method)
    X_val1[pos] = X1
    X_val2[pos] = X2
    y_val[pos] = validation['overlap'][index]
    pos += 1

X_test1 = np.ndarray(shape=(len(test),96,96,n_ch))
X_test2 = np.ndarray(shape=(len(test),96,96,n_ch))
y_test = np.ndarray(shape=(len(test),1))

pos = 0
for index in test.index:
    img1 = np.load(s2mtcp_target + test['pair1'][index])
    img2 = np.load(s2mtcp_target + test['pair2'][index])
    X1 = create_rgb(img1,channel)
    X2 = create_rgb(img2, channel)
    X1 = feature_scaling(X1, method)
    X2 = feature_scaling(X2, method)
    X_test1[pos] = X1
    X_test2[pos] = X2
    y_test[pos] = test['overlap'][index]
    pos += 1

NORM = method
SHUFFLE = False
BATCH_SIZE = 5
dropout = 0.1
decay = 0.0001
model = pretext_task_one_aspp(dropout, decay, 96,96,n_ch)
model.summary()

#Load saved model
#model_name='/saved_models/pretext_tasks/model_pretext1_unclouded_results.h5'
#model.load_weights(model_name)
#######
## Callbacks
callbacks = myCallback()

LEARNING_RATE = 0.001
EPOCHS = 6
optimizer= Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# Record start time
start_time = time.time()

history = model.fit(
    [X_train1, X_train2],
    y_train,
    batch_size = BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=([X_val1, X_val2], y_val),
    callbacks=[callbacks]
)

# Record end time
end_time = time.time()

elapsed_time = end_time - start_time
elapsed_time_minutes = elapsed_time / 60

print(f"Training time: {elapsed_time_minutes:.2f} minutes")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print("Evaluate on val data")
results_val = model.evaluate([X_val1, X_val2], y_val)
print("val loss, val acc:", results_val)
print("Evaluate on test data")
results_test = model.evaluate([X_test1, X_test2], y_test)
print("test loss, test acc:", results_test)
print("Evaluate on train data")
results_train = model.evaluate([X_train1, X_train2], y_train, batch_size=5)
print("train loss, train acc:", results_train)

from sklearn.metrics import confusion_matrix
yv_pred = model.predict([X_test1, X_test2])
for i in range(len(yv_pred)):
    if yv_pred[i] < 0.5:
        yv_pred[i] = 0
    else:
        yv_pred[i]= 1
confusion_matrix(y_test, yv_pred, labels=[0,1])

#visualize predictions 
def scaleMinMax(x):
    return ((x - np.nanpercentile(x,2)) / (np.nanpercentile(x,98) - np.nanpercentile(x,2)))

def create_rgb(x, channel):
    if channel == 'red':
        r = x[:,:,2]
        r = scaleMinMax(r)
        return r
    if channel == 'green':
        g = x[:,:,1]
        g = scaleMinMax(g)
        return g
    if channel == 'blue':
        b  = x[:,:,0]
        b = scaleMinMax(b)
        return b
    if channel == 'rgb':
        r = x[:,:,2]
        g = x[:,:,1]
        b  = x[:,:,0]
        r = scaleMinMax(r)
        g = scaleMinMax(g)
        b = scaleMinMax(b)
        rgb = np.dstack((r,g,b))
        return(rgb)
    
import random
fig, ax = plt.subplots(2, 3, figsize=(8,4),constrained_layout=True)
pair_pos = random.randint(0, len(y_test))
pair1 = tf.concat([create_rgb(X_test1[pair_pos], channel),create_rgb(X_test2[pair_pos],channel)], axis=1)
ax[0,0].imshow(pair1)
ax[0,0].set_title("True: {} | Pred: {}".format(y_test[pair_pos], yv_pred[pair_pos]))
ax[0,0].axis('off')
pair_pos = random.randint(0, len(y_test))
pair2 = tf.concat([create_rgb(X_test1[pair_pos],channel),create_rgb(X_test2[pair_pos],channel)], axis=1)
ax[0,1].imshow(pair2)
ax[0,1].set_title("True: {} | Pred: {}".format(y_test[pair_pos], yv_pred[pair_pos]))
ax[0,1].axis('off')
pair_pos = random.randint(0, len(y_test))
pair3 = tf.concat([create_rgb(X_test1[pair_pos],channel),create_rgb(X_test2[pair_pos],channel)], axis=1)
ax[0,2].imshow(pair3)
ax[0,2].set_title("True: {} | Pred: {}".format(y_test[pair_pos], yv_pred[pair_pos]))
ax[0,2].axis('off')
pair_pos = random.randint(0, len(y_test))
pair1 = tf.concat([create_rgb(X_test1[pair_pos], channel),create_rgb(X_test2[pair_pos],channel)], axis=1)
ax[1,0].imshow(pair1)
ax[1,0].set_title("True: {} | Pred: {}".format(y_test[pair_pos], yv_pred[pair_pos]))
ax[1,0].axis('off')
pair_pos = random.randint(0, len(y_test))
pair2 = tf.concat([create_rgb(X_test1[pair_pos],channel),create_rgb(X_test2[pair_pos],channel)], axis=1)
ax[1,1].imshow(pair2)
ax[1,1].set_title("True: {} | Pred: {}".format(y_test[pair_pos], yv_pred[pair_pos]))
ax[1,1].axis('off')
pair_pos = random.randint(0, len(y_test))
pair3 = tf.concat([create_rgb(X_test1[pair_pos],channel),create_rgb(X_test2[pair_pos],channel)], axis=1)
ax[1,2].imshow(pair3)
ax[1,2].set_title("True: {} | Pred: {}".format(y_test[pair_pos], yv_pred[pair_pos]))
ax[1,2].axis('off')

from datetime import date

today = date.today()
print("Today's date:", today)

str(today)

# #Save our model
# model_name = "8epoch_pre_1_normtrue_unclouded_nodatagen_drop015_callback007_norgb_seed1.h5"
# model.save(model_name)
# print("Saved model to disk")
model_id = generate_short_id()
#save weights / oi random onomasies prepei na figoun 
model_path = '/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/saved_models/'
model_name="Pretext_1_NoPool_"+"S2MTCP_"+model_id+'.h5'
model1 = os.path.join(model_path, model_name)
model.save_weights(model1) 

log_params('S2MTCP',model_id, model_name, LEARNING_RATE, 'Adam', 'binary_crossentropy',EPOCHS, BATCH_SIZE, len(df), len(train), len(test), len(validation), results_train[1], results_train[0], results_val[1], results_val[0], results_test[1], results_test[0], NORM,elapsed_time_minutes)
df_params = pd.read_csv('/home/dvalsamis/Documents/projects/Change_detection_SSL_Siamese/training/pretext_tasks/pretext_task_one_models.csv')

########################## Test the saved weights #######################
model_2 = pretext_task_one_nopool(dropout,decay,96,96,n_ch)
model_2.load_weights(model1)
LEARNING_RATE = 0.001
optimizer= Adam(learning_rate=LEARNING_RATE)
model_2.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("Evaluate on val data")
results_val = model_2.evaluate([X_val1, X_val2], y_val)
print("val loss, val acc:", results_val)
print("Evaluate on test data")
results_test = model_2.evaluate([X_test1, X_test2], y_test)
print("test loss, test acc:", results_test)
print("Evaluate on train data")
results_train = model_2.evaluate([X_train1, X_train2], y_train, batch_size=5)
print("train loss, train acc:", results_train)

from sklearn.metrics import confusion_matrix
yv_pred = model_2.predict([X_test1, X_test2])
for i in range(len(yv_pred)):
    if yv_pred[i] < 0.5:
        yv_pred[i] = 0
    else:
        yv_pred[i]= 1
confusion_matrix(y_test, yv_pred, labels=[0,1])

##############################
# testing on Onera
onera_pretext_target = '/home/aleoikon/Documents/data/ssl/onera_npys/patches/task1/'
onera_df = pd.read_csv(onera_pretext_target+'dataset.csv', dtype=str)

X_on1 = np.ndarray(shape=(len(onera_df),96,96,n_ch))
X_on2 = np.ndarray(shape=(len(onera_df),96,96,n_ch))
y_on = np.ndarray(shape=(len(onera_df),1))

print(X_on1.shape)
print(X_on2.shape)
print(y_on.shape)

def create_rgb_onera(x,channel):
    if channel == 'red':
        r = x[:,:,2]
        r = np.expand_dims(r, axis=2)
        return r
    if channel == 'green':
        g = x[:,:,1]
        g = np.expand_dims(g, axis=2)
        return g
    if channel == 'blue':
        b = x[:,:,0]
        b = np.expand_dims(b, axis=2)
        return b
    if channel == 'rgb':
        r = x[:,:,0]
        g = x[:,:,1]
        b = x[:,:,2]
        rgb = np.dstack((r,g,b))
        return(rgb)     
    else:
        return x
        
img1 = np.load(onera_pretext_target + onera_df['pair1'][0])
 
pos = 0
for index in onera_df.index:
    print(index)
    img1 = np.load(onera_pretext_target + onera_df['pair1'][index])
    img2 = np.load(onera_pretext_target + onera_df['pair2'][index])
    X1 = create_rgb_onera(img1, channel)
    X2 = create_rgb_onera(img2, channel)
    X1 = (X1 - X1.mean()) / X1.std()
    X2 = (X2 - X2.mean()) / X2.std()
    X_on1[pos] = X1
    X_on2[pos] = X2
    y_on[pos] = onera_df['overlap'][index]
    pos += 1
print("Evaluate on Onera data")
results = model.evaluate([X_on1, X_on2], y_on, batch_size=5)
print("loss, acc:", results)

y_on_pred = model.predict([X_on1, X_on2])
for i in range(len(y_on_pred)):
    if y_on_pred[i] < 0.5:
        y_on_pred[i] = 0
    else:
        y_on_pred[i]= 1    

confusion_matrix(y_on, y_on_pred, labels=[0,1])

log_params('Onera', model_id, model_name, LEARNING_RATE, 'Adam', 'binary_crossentropy',EPOCHS, BATCH_SIZE, len(onera_df), 0, 0, 0, results[1], results[0], 0, 0, 0, 0, NORM,elapsed_time_minutes)


import random
fig, ax = plt.subplots(2, 3, figsize=(8,4),constrained_layout=True)
pair_pos = random.randint(0, len(y_on))
pair1 = tf.concat([create_rgb(X_on1[pair_pos], channel),create_rgb(X_on2[pair_pos], channel)], axis=1)
ax[0,0].imshow(pair1)
ax[0,0].set_title("True: {} | Pred: {}".format(y_on[pair_pos], y_on_pred[pair_pos]))
ax[0,0].axis('off')
pair_pos = random.randint(0, len(y_on))
pair2 = tf.concat([create_rgb(X_on1[pair_pos ], channel),create_rgb(X_on2[pair_pos ], channel)], axis=1)
ax[0,1].imshow(pair2)
ax[0,1].set_title("True: {} | Pred: {}".format(y_on[pair_pos], y_on_pred[pair_pos]))
ax[0,1].axis('off')
pair_pos = random.randint(0, len(y_on))
pair3 = tf.concat([create_rgb(X_on1[pair_pos], channel),create_rgb(X_on2[pair_pos], channel)], axis=1)
ax[0,2].imshow(pair3)
ax[0,2].set_title("True: {} | Pred: {}".format(y_on[pair_pos], y_on_pred[pair_pos]))
ax[0,2].axis('off')
pair_pos = random.randint(0, len(y_on))
pair1 = tf.concat([create_rgb(X_on1[pair_pos], channel),create_rgb(X_on2[pair_pos], channel)], axis=1)
ax[1,0].imshow(pair1)
ax[1,0].set_title("True: {} | Pred: {}".format(y_on[pair_pos], y_on_pred[pair_pos]))
ax[1,0].axis('off')
pair_pos = random.randint(0, len(y_on))
pair2 = tf.concat([create_rgb(X_on1[pair_pos ], channel),create_rgb(X_on2[pair_pos ], channel)], axis=1)
ax[1,1].imshow(pair2)
ax[1,1].set_title("True: {} | Pred: {}".format(y_on[pair_pos], y_on_pred[pair_pos]))
ax[1,1].axis('off')
pair_pos = random.randint(0, len(y_on))
pair3 = tf.concat([create_rgb(X_on1[pair_pos], channel),create_rgb(X_on2[pair_pos], channel)], axis=1)
ax[1,2].imshow(pair3)
ax[1,2].set_title("True: {} | Pred: {}".format(y_on[pair_pos], y_on_pred[pair_pos]))
ax[1,2].axis('off')

print("Finished")