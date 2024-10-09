import serial
import numpy as np
import tensorflow as tf
import os
# os.chdir('C:\\Users\\aryan\\OneDrive\\Desktop\\COE SEM 6\\Capstone\\Project\\Project')
# from preprocess.preprocess_data import *
from pred import *
# from .preprocess_data import *
# from preprocess.preprocess_data import *




# Load trained model
model = tf.keras.models.load_model('dqn_wheelchair_model.h5')

# Connect to Arduino
ser_input = serial.Serial('COM3', 9600)
# ser_output = serial.Serial('COM3', 9600)

eeg_data = { 'C3': [], 'Cz': [], 'C4': [] }
# eeg_data = { 'Cz':[]}

def send_command(command):
    print(command.encode())
    # ser_output.write(command.encode())

while True:
    if ser_input.in_waiting > 0:
        line = ser_input.readline().decode().strip()
        signalC3, signalCz, signalC4 = map(int, line.split(","))
        eeg_data['C3'].append(signalC3)
        eeg_data['Cz'].append(signalCz)
        eeg_data['C4'].append(signalC4)
        # eeg_data['Cz'].append(signalCz)

        if len(eeg_data['Cz']) >= 250:
            data_array = [ np.array(eeg_data['Cz'])]
            features = extract_features(data_array)
            features = np.reshape(features, [1, -1])  # Reshape for model input

            prediction = model.predict(features)
            action = np.argmax(prediction[0])

            if action == 1:
                send_command('F')  # Forward
            else:
                send_command('B')  # Backward

            eeg_data['C3'] = eeg_data['C3'][1:]  
            eeg_data['Cz'] = eeg_data['Cz'][1:]
            eeg_data['C4'] = eeg_data['C4'][1:]
            # eeg_data['Cz'] = eeg_data['Cz'][1:]