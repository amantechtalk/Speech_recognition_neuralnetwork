from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional,Flatten, SpatialDropout1D,SpatialDropout2D,SpatialDropout3D
from python_speech_features import mfcc
from python_speech_features import logfbank
from keras.models import Sequential, load_model
from keras.optimizers import adam_v2
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder,normalize
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import scipy.io.wavfile as wav
import numpy as np
import keras
import csv
import os
import time

#Calculating x_test and y_test
scaler = MinMaxScaler(feature_range=(0,1))
test_labels = []
test_data = []

#test_labels.txt is a txt file with all labels for the speech samples that is required for the evaluation. We loop through it to calculate the MFCC value for each speech sample and then normalize it 
with open('C:/Users/amank/Downloads/audiodetect/bed/test.csv', newline='') as tsvfile:
    reader = csv.DictReader(tsvfile)
    
    reader = csv.reader(tsvfile, delimiter=',')
    
    for row in reader:
        
        wav_file = "C:/Users/amank/Downloads/audiodetect/bed/" + row[0]
        (rate,sig) = wav.read(wav_file)
       
        #Getting the MFCC value from the .wav files.
        if(int(sig.shape[0])>=70000):
            mfcc_feat = mfcc(sig[45000:55000],rate)      
        elif(int(sig.shape[0])>=60000):
            mfcc_feat = mfcc(sig[45000:55000],rate)
        elif(int(sig.shape[0])>=50000):
            mfcc_feat = mfcc(sig[30000:40000],rate)            
  
        elif(int(sig.shape[0])>=40000):
            mfcc_feat = mfcc(sig[25000:35000],rate)


        elif(int(sig.shape[0])>=30000):
            mfcc_feat = mfcc(sig[10000:20000],rate)
        elif(int(sig.shape[0])>=20000):
            mfcc_feat = mfcc(sig[5000:15000],rate)            
  
        elif(int(sig.shape[0])>=10000):
            mfcc_feat = mfcc(sig[1000:11000],rate)
        else:
            mfcc_feat = mfcc(sig[1000:11000],rate)
         
        scaler = scaler.fit(mfcc_feat)

        #Normalizing the MFCC values.
        normalized = scaler.transform(mfcc_feat)
        normalized=np.hstack((normalized,normalized))
        normalized=np.hstack((normalized,normalized))
        if(normalized.shape[0]==61) : 
         test_data.append(normalized)
         test_labels.append(str(row[1]))
        










with open('C:/Users/amank/Downloads/audiodetect/cat/test.csv', newline='') as tsvfile:
    reader = csv.DictReader(tsvfile)
    reader = csv.reader(tsvfile, delimiter=',')
    for row in reader:
        wav_file = "C:/Users/amank/Downloads/audiodetect/cat/" + row[0]
        (rate,sig) = wav.read(wav_file)
       
        #Getting the MFCC value from the .wav files.
        if(int(sig.shape[0])>=70000):
            mfcc_feat = mfcc(sig[45000:55000],rate)      
        elif(int(sig.shape[0])>=60000):
            mfcc_feat = mfcc(sig[45000:55000],rate)
        elif(int(sig.shape[0])>=50000):
            mfcc_feat = mfcc(sig[30000:40000],rate)            
  
        elif(int(sig.shape[0])>=40000):
            mfcc_feat = mfcc(sig[25000:35000],rate)


        elif(int(sig.shape[0])>=30000):
            mfcc_feat = mfcc(sig[10000:20000],rate)
        elif(int(sig.shape[0])>=20000):
            mfcc_feat = mfcc(sig[5000:15000],rate)            
  
        elif(int(sig.shape[0])>=10000):
            mfcc_feat = mfcc(sig[1000:11000],rate)
        else:
            mfcc_feat = mfcc(sig[1000:11000],rate)
         
        scaler = scaler.fit(mfcc_feat)

        #Normalizing the MFCC values.
        normalized = scaler.transform(mfcc_feat)
        normalized=np.hstack((normalized,normalized))
        normalized=np.hstack((normalized,normalized))
        if(normalized.shape[0]==61) : 
         test_data.append(normalized)
         test_labels.append(str(row[1]))
        








with open('C:/Users/amank/Downloads/audiodetect/happy/test.csv', newline='') as tsvfile:
    reader = csv.DictReader(tsvfile)
    reader = csv.reader(tsvfile, delimiter=',')
    for row in reader:
        wav_file = "C:/Users/amank/Downloads/audiodetect/happy/" + row[0]
        (rate,sig) = wav.read(wav_file)
       
        #Getting the MFCC value from the .wav files.
        if(int(sig.shape[0])>=70000):
            mfcc_feat = mfcc(sig[45000:55000],rate)      
        elif(int(sig.shape[0])>=60000):
            mfcc_feat = mfcc(sig[45000:55000],rate)
        elif(int(sig.shape[0])>=50000):
            mfcc_feat = mfcc(sig[30000:40000],rate)            
  
        elif(int(sig.shape[0])>=40000):
            mfcc_feat = mfcc(sig[25000:35000],rate)


        elif(int(sig.shape[0])>=30000):
            mfcc_feat = mfcc(sig[10000:20000],rate)
        elif(int(sig.shape[0])>=20000):
            mfcc_feat = mfcc(sig[5000:15000],rate)            
  
        elif(int(sig.shape[0])>=10000):
            mfcc_feat = mfcc(sig[1000:11000],rate)
        else:
            mfcc_feat = mfcc(sig[1000:11000],rate)
         
        scaler = scaler.fit(mfcc_feat)

        #Normalizing the MFCC values.
        normalized = scaler.transform(mfcc_feat)
        normalized=np.hstack((normalized,normalized))
        normalized=np.hstack((normalized,normalized))
        if(normalized.shape[0]==61) : 
         test_data.append(normalized)
         test_labels.append(str(row[1]))
        
    label_encoder_test = LabelEncoder()
    vec_test = label_encoder_test.fit_transform(test_labels)
   
    #One hot encoding the labels
    one_hot_labels_test = keras.utils.to_categorical(vec_test, num_classes=3)
    Y_test = one_hot_labels_test
    X_test = np.array(test_data,dtype=np.float32)








#Loading x_train
train_labels = []

#train_labels.txt is a txt file with all labels for the speech samples that is required for the evaluation. We loop through it to calculate the MFCC value for each speech sample and then normalize it 
with open('C:/Users/amank/Downloads/audiodetect/bed/train.csv', newline='') as tsvfile:
    reader = csv.DictReader(tsvfile)
    reader = csv.reader(tsvfile, delimiter=',')
    X_train = []
    for row in reader:
        wav_file = "C:/Users/amank/Downloads/audiodetect/bed/" + row[0]
        (rate,sig) = wav.read(wav_file)
        
        #Getting the MFCC value from the .wav files.
       

        if(int(sig.shape[0])>=70000):
            mfcc_feat = mfcc(sig[45000:55000],rate)      
        elif(int(sig.shape[0])>=60000):
            mfcc_feat = mfcc(sig[45000:55000],rate)
        elif(int(sig.shape[0])>=50000):
            mfcc_feat = mfcc(sig[30000:40000],rate)            
  
        elif(int(sig.shape[0])>=40000):
            mfcc_feat = mfcc(sig[25000:35000],rate)


        elif(int(sig.shape[0])>=30000):
            mfcc_feat = mfcc(sig[10000:20000],rate)
        elif(int(sig.shape[0])>=20000):
            mfcc_feat = mfcc(sig[5000:15000],rate)            
  
        elif(int(sig.shape[0])>=10000):
            mfcc_feat = mfcc(sig[1000:11000],rate)
        else:
            mfcc_feat = mfcc(sig[1000:11000],rate)
              
        scaler = scaler.fit(mfcc_feat)
        
        normalized = scaler.transform(mfcc_feat)
        normalized=np.hstack((normalized,normalized))
        normalized=np.hstack((normalized,normalized))
        if(normalized.shape[0]==61) : 
         X_train.append(np.array(normalized, dtype=np.float32))
         train_labels.append(str(row[1]))
with open('C:/Users/amank/Downloads/audiodetect/cat/train.csv', newline='') as tsvfile:
    reader = csv.DictReader(tsvfile)
    reader = csv.reader(tsvfile, delimiter=',')
    
    for row in reader:
        wav_file = "C:/Users/amank/Downloads/audiodetect/cat/" + row[0]
        (rate,sig) = wav.read(wav_file)
        
        #Getting the MFCC value from the .wav files.
       

        if(int(sig.shape[0])>=70000):
            mfcc_feat = mfcc(sig[45000:55000],rate)      
        elif(int(sig.shape[0])>=60000):
            mfcc_feat = mfcc(sig[45000:55000],rate)
        elif(int(sig.shape[0])>=50000):
            mfcc_feat = mfcc(sig[30000:40000],rate)            
  
        elif(int(sig.shape[0])>=40000):
            mfcc_feat = mfcc(sig[25000:35000],rate)


        elif(int(sig.shape[0])>=30000):
            mfcc_feat = mfcc(sig[10000:20000],rate)
        elif(int(sig.shape[0])>=20000):
            mfcc_feat = mfcc(sig[5000:15000],rate)            
  
        elif(int(sig.shape[0])>=10000):
            mfcc_feat = mfcc(sig[1000:11000],rate)
        else:
            mfcc_feat = mfcc(sig[1000:11000],rate)
              
        scaler = scaler.fit(mfcc_feat)
        
        normalized = scaler.transform(mfcc_feat)
        normalized=np.hstack((normalized,normalized))
        normalized=np.hstack((normalized,normalized))
        if(normalized.shape[0]==61) : 
         X_train.append(np.array(normalized, dtype=np.float32))
         train_labels.append(str(row[1]))

with open('C:/Users/amank/Downloads/audiodetect/happy/train.csv', newline='') as tsvfile:
    reader = csv.DictReader(tsvfile)
    reader = csv.reader(tsvfile, delimiter=',')
    
    for row in reader:
        wav_file = "C:/Users/amank/Downloads/audiodetect/happy/" + row[0]
        (rate,sig) = wav.read(wav_file)
        
        #Getting the MFCC value from the .wav files.
       

        if(int(sig.shape[0])>=70000):
            mfcc_feat = mfcc(sig[45000:55000],rate)      
        elif(int(sig.shape[0])>=60000):
            mfcc_feat = mfcc(sig[45000:55000],rate)
        elif(int(sig.shape[0])>=50000):
            mfcc_feat = mfcc(sig[30000:40000],rate)            
  
        elif(int(sig.shape[0])>=40000):
            mfcc_feat = mfcc(sig[25000:35000],rate)


        elif(int(sig.shape[0])>=30000):
            mfcc_feat = mfcc(sig[10000:20000],rate)
        elif(int(sig.shape[0])>=20000):
            mfcc_feat = mfcc(sig[5000:15000],rate)            
  
        elif(int(sig.shape[0])>=10000):
            mfcc_feat = mfcc(sig[1000:11000],rate)
        else:
            mfcc_feat = mfcc(sig[1000:11000],rate)
              
        scaler = scaler.fit(mfcc_feat)
        
        normalized = scaler.transform(mfcc_feat)
        normalized=np.hstack((normalized,normalized))
        normalized=np.hstack((normalized,normalized))
        if(normalized.shape[0]==61) : 
         X_train.append(np.array(normalized, dtype=np.float32))
         train_labels.append(str(row[1]))



         
print(rate,sig,mfcc_feat,normalized)

X_train_array = np.array(X_train)


#calculate y_train
label_encoder = LabelEncoder()
vec = label_encoder.fit_transform(train_labels)
one_hot_labels_train = keras.utils.np_utils.to_categorical(vec, num_classes=3)
Y_train = one_hot_labels_train
print(X_train_array.shape,Y_train.shape, X_test.shape,Y_test.shape)

time.sleep(10)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D





model_11 = Sequential()
model_11.add(LSTM(200,input_shape=(61,52),return_sequences=False))
model_11.add(Dense(3,activation='softmax'))
model_11.summary()

model_11.compile(optimizer=Adam(amsgrad=True, lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
EPOCHS = 20
BATCH_SIZE = 40

#Training the model
history = model_11.fit(X_train_array, 
                    Y_train, 
                    epochs=EPOCHS,
                   # callbacks = callbacks,
                    batch_size=BATCH_SIZE,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    shuffle=True
                   )


#plotting the loss
history = model_11.history
#print(history.history['loss'])
#print(history.history['val_loss'])
pyplot.plot(history.history['loss'])
pyplot.plot(history.history["val_loss"])
pyplot.title("train and validation loss")
pyplot.ylabel("value")
pyplot.xlabel("epoch")
pyplot.legend(['train','validation'])
pyplot.show()
#plotting the loss
history = model_11.history
#print(history.history['loss'])
#print(history.history['val_loss'])
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history["val_accuracy"])
pyplot.title("train and validation accuracy")
pyplot.ylabel("accuracy value")
pyplot.xlabel("epoch")
pyplot.legend(['train','validation'])
pyplot.show()
scores = model_11.evaluate(X_test, Y_test, verbose=1)


model_11.save_weights('model.h5')






