import numpy as np
import matplotlib.pyplot as plt

#------------NORMAL VOLTAGE SIGNAL----------------
frequency_hz = 60
sample_rate_hz = 6000 
duration_seconds = 0.2
amplitude = 1.0 #To control the wave height

number_of_samples = int(sample_rate_hz * duration_seconds)
time = np.arange(number_of_samples) / sample_rate_hz #This is to know at what time each measurement is taken
voltage = amplitude * np.sin(2 * np.pi * frequency_hz * time) #This is the formula for a sine wave.creates the wave’s value at every timestamp stored in time
#v(t) = A*sin(2*pi*f*t + phi)

# plt.plot(time, voltage)
# plt.xlabel('Time (s)')
# plt.ylabel('Relative Amplitude (V)')
# plt.title('Normal Voltage signal - 60Hz')
# plt.show()

#------------VOLTAGE SAG------------------
#A voltage sag is a brief drop in the size or magnitude of the voltage, followed by a return to normal
sag_voltage = voltage.copy()
sag_period = (time >= 0.05) & (time < 0.15) #This is to create a sag between 0.05 and 0.1 seconds
sag_voltage[sag_period] = 0.5 * sag_voltage[sag_period] #Reduce the voltage to 50% during the sag period

# plt.plot(time, voltage, label='Normal Voltage', linestyle='--')
# plt.plot(time, sag_voltage, label='Voltage Sag', color='red')
# plt.xlabel('Time (s)')
# plt.ylabel('Relative Amplitude (V)')
# plt.title('Normal Signal compared with Voltage Sag - 60Hz')
# plt.legend()
# plt.show()

#-----------VOLTAGE SWELL------------------
#A voltage swell is a brief increase in the size or magnitude of the voltage, followed by a return to normal
swell_voltage = voltage.copy() #Making a copy of the original voltage signal to create a swell
swell_period = (time >= 0.05) & (time < 0.15) #This is to create a swell between 0.05 and 0.1 seconds
swell_voltage[swell_period] = 1.5 * swell_voltage[swell_period] #Increase in the amplitude to be 150% of the original voltage during the swell period. A 50% increase in the voltage is considered a swell.

# plt.plot(time, voltage, label='Normal Voltage', linestyle='--')
# plt.plot(time, swell_voltage, label='Voltage Swell', color='green')
# plt.xlabel('Time (s)')
# plt.ylabel('Relative Amplitude (V)')
# plt.title('Normal Signal compared with Voltage Swell - 60Hz')
# plt.legend()
# plt.show()

#-----------VOLTAGE INTERRUPTION------------------
#A voltage interruption is a complete loss of voltage for a brief period of time, followed by a return to normal
interruption_voltage = voltage.copy() #Making a copy of the original voltage signal to create an interruption
interruption_period = (time >= 0.05) & (time < 0.15) #This is to create an interruption between 0.05 and 0.1 seconds
interruption_voltage[interruption_period] = 0 #Set the voltage to zero during the interruption period

# Sag:          the wave continues at half its normal amplitude.
# Interruption: the wave becomes zero in this example.

# plt.plot(time, voltage, label='Normal Voltage', linestyle='--')
# plt.plot(time, sag_voltage, label='Voltage Sag', color='red')
# plt.plot(time, swell_voltage, label='Voltage Swell', color='green')
# plt.plot(time, interruption_voltage, label='Voltage Interruption', color='purple')
# plt.xlabel('Time (s)')
# plt.ylabel('Relative Amplitude (V)')
# plt.title('Normal Signal compared with Voltage Sag, Swell, and Interruption - 60Hz')
# plt.ylim(-1.7,1.7) #fixes the vertical scale so the graphs are easier to compare.
# plt.legend()
# plt.show()

#-----------HARMONIC DISTORTION------------------
#Harmonic distortion is a deviation from the ideal sinusoidal waveform, caused by the presence
# of harmonics in the signal. Harmonics are integer multiples of the fundamental frequency.
third_harmonic = 0.2 * amplitude * np.sin(2 * np.pi * 3 * frequency_hz * time) #Third harmonic is 3 times the fundamental frequency
fifth_harmonic = 0.1 * amplitude * np.sin(2 * np.pi * 5 * frequency_hz * time) #Fifth harmonic is 5 times the fundamental frequency
harmonic_voltage = voltage + third_harmonic + fifth_harmonic

# plt.plot(time, voltage, label='Normal Voltage', linestyle='--')
# plt.plot(time, harmonic_voltage, label='Harmonic Distortion', color='orange')
# plt.xlabel('Time (s)')
# plt.ylabel('Relative Amplitude')
# plt.title('Normal Signal compared with Harmonic Distortion - 60Hz')
# plt.ylim(-1.7,1.7) #fixes the vertical scale so the graphs are easier to compare.
# plt.xlim(0,0.05) #displays only the first 0.05 seconds, making the shape easier to examine. It doesn’t remove any data.
# plt.legend()
# plt.show()  

#-----------SAVING THE SIGNAL AS A CSV FILE------------------
#This is to save the generated signals as a CSV file for further analysis or use in other
# applications. The CSV file will contain two columns: time and voltage.
normal_date = np.column_stack((time, voltage))

np.savetxt("normal_signal.csv", normal_date, delimiter=",", header="time_seconds, voltage_relative", comments="")





