import matplotlib.pyplot as plt
from numpy import genfromtxt


def read_file(file_name):
    full_name = file_name + ".csv"
    data = genfromtxt(full_name, delimiter=',')
    return data

pos_loss = read_file('pos_loss')
vel_loss = read_file('vel_loss')
plt.figure(1)
plt.plot(pos_loss[1], '-', color='red', label='lane1')
plt.show()

plt.figure(2)
plt.plot(vel_loss[1], '-', color='blue', label='lane2')
plt.show()