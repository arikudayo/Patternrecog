# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:58:34 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

# Generate time vector
t = np.linspace(0, 1, 1000)

# Generate sine and cosine signals
s1 = np.sin(2 * np.pi * 5 * t)  # Sine signal with period of 0.2
s2 = np.cos(2 * np.pi * 20 * t)  # Cosine signal with period of 0.1

# Mix signals
S = np.c_[s1, s2]  # 2D array of signals
A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
X = S.dot(A.T)  # Mixed signals

# Separate signals using ICA
ica = FastICA(n_components=2)
S_ = ica.fit_transform(X)  # Separated signals
A_ = ica.mixing_  # Estimated mixing matrix

# Plotting
plt.figure(figsize=(12, 8))

# Mixed signals
plt.subplot(3, 1, 1)
plt.title('Mixed Signals')
plt.plot(X)
plt.legend(['Mixed Signal 1', 'Mixed Signal 2'])

# Original signals
plt.subplot(3, 1, 2)
plt.title('Original Signals')
plt.plot(S)
plt.legend(['Sine Signal', 'Cosine Signal'])

# Separated signals
plt.subplot(3, 1, 3)
plt.title('Separated Signals (ICA)')
plt.plot(S_)
plt.legend(['Separated Signal 1', 'Separated Signal 2'])

plt.tight_layout()
plt.show()