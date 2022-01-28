#
# Visualization script for the 3-body example for the dcEmb package
# 
# Copyright (C) 2022 Embecosm Limited
# 
# Contributor William Jones <william.jones@embecosm.com>
# Contributor Elliot Stein <E.Stein@soton.ac.uk>
# 
# This file is part of the dcEmb package
# 
# SPDX-License-Identifier: GPL-3.0-or-later 
# 

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import pandas as pd

fig = plt.figure()
ax = fig.add_subplot()

def update(num, data_true, data_deriv, data_org,
    dot1, dot2, dot3,
    dot4, dot5, dot6,
    dot7, dot8, dot9):
    ax.set_title(num)
    dot1.set_data(data_true[1, num], data_true[2, num])
    dot2.set_data(data_true[8, num], data_true[9, num])
    dot3.set_data(data_true[15, num], data_true[16, num])
    dot4.set_data(data_deriv[1, num], data_deriv[2, num])
    dot5.set_data(data_deriv[8, num], data_deriv[9, num])
    dot6.set_data(data_deriv[15, num], data_deriv[16, num])
    dot7.set_data(data_org[1, num], data_org[2, num])
    dot8.set_data(data_org[8, num], data_org[9, num])
    dot9.set_data(data_org[15, num], data_org[16, num])
    return dot1, dot2, dot3, dot4, dot5, dot6, dot7, dot8, dot9

df_true=pd.read_csv('true_generative.csv', sep=',',header=None)
df_deriv=pd.read_csv('deriv_generative.csv', sep=',',header=None)
df_org=pd.read_csv('org_generative.csv', sep=',',header=None)
N = (df_true.shape)[0]
skip = N//1000

data_true = df_true.values[0::skip].T
data_deriv = df_deriv.values[0::skip].T
data_org = df_org.values[0::skip].T

line1, = ax.plot(data_true[1, 0:], data_true[2, 0:], '-', color='#e0e0e0')
dot1, = ax.plot(data_true[1, 0:1], data_true[2, 0:1], 'o', color='#b71c1c')
dot2, = ax.plot(data_true[8, 0:1], data_true[9, 0:1], 'o', color='#1b5e20')
dot3, = ax.plot(data_true[15, 0:1], data_true[16, 0:1], 'o', color='#0d47a1')

dot4, = ax.plot(data_deriv[1, 0:1], data_deriv[2, 0:1], 'o', color='#e53935')
dot5, = ax.plot(data_deriv[8, 0:1], data_deriv[9, 0:1], 'o', color='#43a047')
dot6, = ax.plot(data_deriv[15, 0:1], data_deriv[16, 0:1], 'o', color='#1e88e5')

dot7, = ax.plot(data_org[1, 0:1], data_org[2, 0:1], 'o', color='#e57373')
dot8, = ax.plot(data_org[8, 0:1], data_org[9, 0:1], 'o', color='#81c784')
dot9, = ax.plot(data_org[15, 0:1], data_org[16, 0:1], 'o', color='#64b5f6')

ma = data_org[[1,2,8,9,15,16],].max()

# Setting the axes properties
ax.set_xlim([-ma, ma])
ax.set_xlabel('X')

ax.set_ylim([-ma, ma])
ax.set_ylabel('Y')

ani = animation.FuncAnimation(fig, update, N//skip, fargs=(
    data_true, data_deriv, data_org,
    dot1, dot2, dot3,
    dot4, dot5, dot6,
    dot7, dot8, dot9), interval=1, blit=False)
plt.show()
ani.save('./dynamic.mp4', fps=60, dpi=300)
