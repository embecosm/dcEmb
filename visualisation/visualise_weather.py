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

def update(num, ax, data_true, data_deriv, data_org,
    line1, line2, line3, fill2, fill3):
    ax.set_title(num)
    x_vals = list(range(0, num+1))
    line1.set_data(x_vals, data_true[25, 0:num+1])
    line2.set_data(x_vals, data_deriv[25, 0:num+1])
    line3.set_data(x_vals, data_org[25, 0:num+1])
    ax.collections.clear()
    fill2, = ax.fill_between(x_vals, data_deriv[25, 0:num+1] - data_deriv_c[25,0:num+1], data_deriv[25, 0:num+1] + data_deriv_c[25, 0:num+1], color='#1b5e20', alpha=0.25),
    fill3, = ax.fill_between(x_vals, data_org[25, 0:num+1] - data_org_c[25, 0:num+1], data_org[25, 0:num+1] + data_org_c[25, 0:num+1], color='#0d47a1', alpha=0.25), 
    return line1, line2, line3, fill2, fill3

df_true=pd.read_csv('weather/true_generative.csv', sep=',',header=None)
df_deriv=pd.read_csv('weather/pos_generative.csv', sep=',',header=None)
df_org=pd.read_csv('weather/prior_generative.csv', sep=',',header=None)
df_org_c=pd.read_csv('weather/prior_generative_var.csv', sep=',',header=None)
df_deriv_c=pd.read_csv('weather/pos_generative_var.csv', sep=',',header=None)
N = (df_true.shape)[0]
skip = N//1000


data_true = df_true.values.T
data_deriv = df_deriv.values.T
data_org = df_org.values.T
data_org_c = df_org_c.values.T
data_deriv_c = df_deriv_c.values.T

line1, = ax.plot(0, data_true[25, 0:1], '-', color='#b71c1c', label='true')
line2, = ax.plot(0, data_deriv[25, 0:1], '-', color='#1b5e20', label='posterior')
line3, = ax.plot(0, data_org[25, 0:1], '-', color='#0d47a1', label='prior')
fill2, = ax.fill_between(0, data_deriv[25, 0:1] - data_deriv_c[25, 0:1], data_deriv[25, 0:1] + data_deriv_c[25, 0:1], color='#1b5e20', alpha=0.25),
fill3, = ax.fill_between(0, data_org[25, 0:1] - data_org_c[25, 0:1], data_org[25, 0:1] + data_org_c[25, 0:1], color='#0d47a1', alpha=0.25), 

# dot1, = ax.plot(data_true[1, 0:1], data_true[2, 0:1], 'o', color='#b71c1c')
# dot2, = ax.plot(data_true[8, 0:1], data_true[9, 0:1], 'o', color='#1b5e20')
# dot3, = ax.plot(data_true[15, 0:1], data_true[16, 0:1], 'o', color='#0d47a1')

# dot4, = ax.plot(data_deriv[1, 0:1], data_deriv[2, 0:1], 'o', color='#e53935')
# dot5, = ax.plot(data_deriv[8, 0:1], data_deriv[9, 0:1], 'o', color='#43a047')
# dot6, = ax.plot(data_deriv[15, 0:1], data_deriv[16, 0:1], 'o', color='#1e88e5')

# dot7, = ax.plot(data_org[1, 0:1], data_org[2, 0:1], 'o', color='#e57373')
# dot8, = ax.plot(data_org[8, 0:1], data_org[9, 0:1], 'o', color='#81c784')
# dot9, = ax.plot(data_org[15, 0:1], data_org[16, 0:1], 'o', color='#64b5f6')

ma = max([data_true[[25],].max(), data_deriv[[25],].max(), data_org[[25],].max()])

# Setting the axes properties
ax.set_xlim([0, N])
ax.set_xlabel('X')

ax.set_ylim([-ma, ma])
ax.set_ylabel('Y')

ax.legend()

ani = animation.FuncAnimation(fig, update, N, fargs=(ax, 
    data_true, data_deriv, data_org,
    line1, line2, line3, fill2, fill3), repeat=False)
plt.show()
ani.save('./weather.mp4', fps=60, dpi=300)
