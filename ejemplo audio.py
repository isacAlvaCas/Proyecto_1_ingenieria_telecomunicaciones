#import numpy as np
#from numpy.fft import fft,ifft
#import matplotlib.pyplot as plt #biblioteca para graficar
import scipy.io.wavfile as waves #biblioteca para procesar audio .wav en python
import scipy.integrate as integrate
from scipy import signal
 #ejemplo procesar audio
import numpy as np
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt #biblioteca para graficar
import scipy.io.wavfile as waves #biblioteca para procesar audio .wav en python
import scipy.integrate as integrate
from scipy import signal
# INGRESO
# archivo = input('archivo de audio: ')
'''
arch=input('ingrese el nombre del archivo: ')
archivo=(arch +'.wav')
'''

archivo = '440Hz_44100Hz_16bit_05sec.wav' #archivo de entrada
# PROCEDIMIENTO
muestreo, sonido = waves.read(archivo) #leer archivo de audio, devuelve la frecuencia de muestreo de datos por segundo y sonido devuelve datosd e sonido, donde los indices devuelven la cantidad de muestras y canal correspondiente
tamano_sonido = len(sonido.shape) #tamano de todas las muestras
dt = 1/muestreo #Periodo de muestreo 
t = np.arange(0,tamano_sonido*dt,dt) #valor de intervalos de tiempo del audio
#procesamiento de canales del audio
#disenar para que selecciones canal monofonico o estereo, seguan sea el caso
canal = 0 #monofonico 
#ruido=100*np.random.random(122868)#senal de ruido blanco
if (tamano_sonido==1): # Monofónico
    uncanal=sonido[:] #+ruido
if (tamano_sonido==2): # Estéreo
    uncanal=sonido[:,canal] #+ruido

#uncanal = sonido[:] #canal monofonico (todas las muestars)

wn= 22050/30000
b, a = signal.butter(8, wn, 'lowpass',analog=True)   #Configuration filter 8 representa el orden del filtro
filtedData = signal.filtfilt(b, a, uncanal)  #data es la señal a filtrar
X = fft(filtedData)
N = len(X)
n = np.arange(N)
T = N*2.26e-05
freq = n/T - 22050



#procedimiento de codificacion delta-sigma

 #senal de sonido con ruido blanco
ganancia_deltaY=0.3
deltaY = ganancia_deltaY*np.max(sonido) #senal deltaY ganancia*maximo de senal real
#deltaT = dt #periodo de senal delta
deltaT = 2.2e-06
i_muestras = int(0/deltaT) #intervalo muestras
tf_muestra=10*0.002 #hasta aqui llega el muestreo
t_muestra=np.arange(0,tf_muestra,deltaT) #tiempo que dura senal de muestra del audio
tamano = len(t_muestra) #
muestra= np.copy(uncanal[:i_muestras+tamano])#solo una muestra de la sanal del canal monofonico

# Señal Digital
vT = np.zeros(tamano, dtype=float)  #vector de valores codificados de la senal real  
bkT = np.zeros(tamano, dtype=int) #vector de bits a la salida de la senal codificada

for k in range(1,tamano):
    #pwm
    dif = muestra[k]-vT[k-1]
    if (dif>0):
        bit = 1
    else:
        bit = -1
    vT[k] = vT[k-1]+bit*deltaY
    bkT[k] = bit
parametros=np.array([deltaT,deltaY,tamano])

# SALIDA
print('Parametros: ')
print(parametros)
print('datos de señal modulada:')
print(bkT)
np.savetxt('deltasigma_parametros.txt',parametros)
np.savetxt('deltasigma_datos.txt',bkT,fmt='%i')
print('... archivos.txt guardados ...')
###

# cuadrado de la señal para el integral de energía
cuadrado = muestra**2
#integrando a partir de muestras de la señal
energia = integrate.simps(cuadrado,t_muestra)
# SALIDA
print(f" Energia de la onda del audio (muestra) : {energia:.2f}")

'''
plt.figure() 
# Senal real en el dominio del tiempo
plt.figure(figsize = (12, 6))
plt.suptitle('Senal real en el dominio del tiempo')
plt.subplot(211)    # grafica de 2x1 y subgrafica 1
plt.ylabel('Senal real (t)')
plt.xlabel('t (s)')
plt.plot(t,sonido[:,0], 'r')
'''
plt.figure() 
plt.figure(figsize = (12, 6))
plt.subplot(211)
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.plot(freq,np.abs(X))


plt.figure() 
# Senal real en el dominio del tiempo
plt.figure(figsize = (12, 6))
plt.suptitle('Senal real en el dominio del tiempo')
plt.subplot(211)    # grafica de 2x1 y subgrafica 1
plt.ylabel('Senal real (t)')
plt.xlabel('t (s)')
plt.plot(t_muestra[:],muestra[:], 'g')

#Graficar

plt.figure()       
plt.figure(figsize = (12, 6))
plt.suptitle('Codificador Delta-Sigma')

plt.subplot(211)    # grafica de 2x1 y subgrafica 1
plt.ylabel('x(t), x[n]')
plt.xlabel('t,k')
plt.plot(t_muestra[0:90],muestra[0:90], 'g')
plt.step(t_muestra[0:90],vT[0:90], where='post',color='m') # Puntos x[n]




plt.subplot(212)    # grafica de 2x1 y subgrafica 2
plt.ylabel('b_k(l)')
plt.xlabel('l')
plt.axis((0,90,-1.1,1.1))
plt.plot(bkT,'bo')     # Puntos b_k(l)
puntos=np.arange(0,90,1)     #eje x para esacalon
plt.step(puntos[0:90],bkT[0:90], where='post')
plt.show()

'''
Decodificacion
'''
#leer datos

bkT_info=np.loadtxt('deltasigma_datos.txt',dtype=int)
datos=np.loadtxt('deltasigma_parametros.txt',dtype=float)








#fft
'''
X = fft(uncanal+ruido)
N = len(X)
n = np.arange(N)
T = N/muestreo
freq = n/T 




#filtro

wn= (2*5000)/(22050)
b, a = signal.butter(8, wn, 'lowpass')   #Configuration filter 8 representa el orden del filtro
filtedData = signal.filtfilt(b, a, uncanal)  #data es la señal a filtrar
X = fft(filtedData)
N = len(X)
n = np.arange(N)
T = N/muestreo
freq = n/T 

#plt.figure(figsize = (12, 6))
#plt.stem(freq, np.abs(X), 'b', \
#         markerfmt=" ", basefmt="-b")

plt.plot(freq,np.abs(X))
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.grid(1)

plt.plot(freq,np.abs(X))
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')
plt.grid(1)
plt.show()
'''