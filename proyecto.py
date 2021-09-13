import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt #biblioteca para graficar
import scipy.io.wavfile as waves #biblioteca para procesar audio .wav en python
import scipy.integrate as integrate
from scipy import signal
import math
# INGRESO
# archivo = input('archivo de audio: ')
'''
arch=input('ingrese el nombre del archivo: ')
archivo=(arch +'.wav')
'''

archivo = 'Alarm01.wav' #archivo de entrada
# PROCEDIMIENTO
muestreo, sonido = waves.read(archivo) #leer archivo de audio, devuelve la frecuencia de muestreo de datos por segundo y sonido devuelve datosd e sonido, donde los indices devuelven la cantidad de muestras y canal correspondiente
tamano_sonido = len(sonido) #tamano de todas las muestras
dt = 1/muestreo #Periodo de muestreo 
t = np.arange(0,tamano_sonido*dt,dt) #valor de intervalos de tiempo del audio
#procesamiento de canales del audio
#disenar para que selecciones canal monofonico o estereo, seguan sea el caso
canal = 0 #monofonico 

uncanal = sonido[:,canal]
'''
if (tamano_sonido==1): # Monofónico
    uncanal=sonido[:] #+ruido
if (tamano_sonido==2): # Estéreo
    uncanal=sonido[:,canal] #+ruido
'''
#tr=np.arange(0,tamano_sonido*dt,2*dt)
ruido=100*np.random.random(len(t))#senal de ruido blanco
# Senal real en el dominio del tiempo
senal_real = uncanal+ruido

X = fft(senal_real)
N = len(X)
n = np.arange(N)
T = N/muestreo
freq = n/T
#filtro (22050)
sos = signal.butter(0, 10000, 'hp', fs=muestreo, output='sos')
filtered = signal.sosfilt(sos, senal_real)

X1 = fft(filtered)
N1 = len(X1)
n1 = np.arange(N1)
T1 = N1/muestreo
freq2 = n1/T1



bits = 8
fs=2*22050
Ts=1/fs
NQ=pow(2,bits)
Q_fs=max(uncanal) #full_scale
rango=[2*Q_fs]/(NQ-1)


vT = np.array([])
minimo=min(uncanal)
k=0


while(k<NQ):
    valor=uncanal[k]
    vT=np.append(vT,round(valor))
    k+=1
    

bkT = np.array([])
'''
l=0
while(l<NQ):
    bkT=np.append(bkT,bin(l))

print(bkT)
'''




plt.figure(1)
plt.suptitle('Senal real en el dominio del tiempo')
plt.plot(t,senal_real)
plt.xlabel('t segundos')
plt.ylabel('sonido(t)')

plt.figure(2)
plt.suptitle('Transformada de Fourier de Senal real en el dominio del tiempo')
plt.plot(freq,abs(X))
plt.xlabel('t segundos')
plt.ylabel('sonido(t)')

plt.figure(3)
plt.suptitle('Senal filtrada')

plt.plot(t, filtered)

plt.figure(4)
plt.suptitle('Transformada de Fourier de Senal real en el dominio del tiempo')
plt.plot(freq2,abs(X1))
plt.xlabel('t segundos')
plt.ylabel('sonido(t)')
plt.show()
