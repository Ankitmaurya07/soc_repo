import librosa
import os
import matplotlib.pyplot as plt 
import numpy as np
import librosa.display

file_path = r'C:\Users\pashu\OneDrive\Desktop\soc\bird_sound.wav'
if os.path.exists(file_path):
    y, sr = librosa.load(file_path)
    time = librosa.get_duration(y=y, sr=sr)
    
    plt.figure(figsize=(10, 2))
    plt.plot(y)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of bird sound')
    #plt.show()

    # Ensure no zero values to avoid log of zero
    y_no_zeros = np.where(y == 0, np.finfo(float).eps, y)
    
    # Convert amplitude to decibels
    z1 = 20 * np.log10(np.abs(y_no_zeros))
    print(f'z1 (log amplitude values): {z1}')

    # Compute STFT
    z2 = librosa.stft(y)
    
    # Convert amplitude to decibels
    z2_db = librosa.amplitude_to_db(np.abs(z2), ref=np.max)
    
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(z2_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(img, format='%+2.0f dB')
    plt.title('Spectrogram of bird sound')
    #plt.show()

    # Zero crossing rate
    zero_crossings = librosa.zero_crossings(y, pad=False)
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    plt.figure(figsize=(14, 5))
    plt.plot(zcr)
    plt.title("Zero Crossing Rate")
    plt.xlabel('Frames')
    plt.ylabel('Rate')
    #plt.show()


# spectral centroids
spectral_centroids= librosa.feature.spectral_centroid(y=y,sr=sr)[0]

# we have the code to plot spectral centroids for u
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.4)
plt.plot(t, librosa.util.normalize(spectral_centroids), color='r')
plt.title('Spectral Centroid')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Centroid')
#plt.show() #follow us, spectral chords)


# MeI-Frequenxy cepstral coefficients(mfccs)
mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)


plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('MFCC')
plt.xlabel('Time')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.show()


