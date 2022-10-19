from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional,Flatten, SpatialDropout1D,SpatialDropout2D,SpatialDropout3D
from python_speech_features import mfcc
from python_speech_features import logfbank
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
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
import matplotlib.pyplot as plt
try:
 os.remove("recording0.wav")
except:
 print('')   
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write


# Sampling frequency
frequency = 16000


duration = 3.0
print("start")
recording = sd.rec(int(duration * frequency),
				samplerate = frequency, channels =1)


sd.wait()

write("recording0.wav", frequency, recording)












#Calculating x_test and y_test
scaler = MinMaxScaler(feature_range=(0,1))
test_labels = []
test_data = []

wav_file = "recording0.wav"

(rate,sig) = wav.read(wav_file)

        #Getting the MFCC value from the .wav files.
mfcc_feat = mfcc(sig[0:10000],rate)
scaler = scaler.fit(mfcc_feat)
print(mfcc_feat)
print(scaler)
        #Normalizing the MFCC values.
normalized = scaler.transform(mfcc_feat)
normalized=np.hstack((normalized,normalized))
normalized=np.hstack((normalized,normalized))
print(normalized)
#attempt with 200 units
model_11 = Sequential()


model_11 = Sequential()
model_11.add(LSTM(200,input_shape=(61,52),return_sequences=False))
model_11.add(Dense(3,activation='softmax'))



EPOCHS = 100
BATCH_SIZE = 40


model_11.load_weights('model.h5')


a ={0:'bed',1:'cat',2:'happy'}


n=np.reshape(normalized, [1, 61, 52])

prediction = model_11.predict(n)
print(prediction)

maxindex = int(np.argmax(prediction))

print(a[maxindex])



