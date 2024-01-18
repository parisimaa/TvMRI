'''
This file containt the following operations:
fft, ifft, conv2D, conv2DT

'''

import numpy as np
from scipy import fft

class SignalProcessor:

	def __init__(self):
		pass

	def to_fourier_domain(self, x):
		return fft.fftshift(fft.fft2(fft.ifftshift(x)))

	def to_image_domain(self, x):
		return fft.ifftshift(fft.ifft2(fft.fftshift(x)))


	# Defining Conv2D to apply gradient operator during process

	'''
	The 2D convolution code has been modified to handle 
	multi-coil data by ensuring that the input shape has 
	three dimensions, with the third dimension representing 
	the number of channels or coils.

	'''
	def conv2d_fft(self, x, h):
	    p0 = x.shape[0] - h.shape[0]
	    p1 = x.shape[1] - h.shape[1]

	    # Initialize an empty result with the same shape as x
	    result = np.zeros_like(x, dtype=np.complex128)

	    for i in range(x.shape[2]):  # Loop through channels
	        x_channel = x[:, :, i]  # Get the current channel of x
	        h_channel = h[:, :, i]  # Get the current channel of h

	        h_pad = np.pad(h_channel, ((p0 // 2, p0 // 2), (p1 // 2, p1 // 2)))
	        Fh = self.to_fourier_domain(h_pad)
	        Fx = self.to_fourier_domain(x_channel)
	        result[:, :, i] = self.to_image_domain(Fx * Fh)

	    return result

	def conv2dT_fft(self, x, h):
	    p0 = x.shape[0] - h.shape[0]
	    p1 = x.shape[1] - h.shape[1]

	    # Initialize an empty result with the same shape as x
	    result = np.zeros_like(x, dtype=np.complex128)

	    for i in range(x.shape[2]):  # Loop through channels
	        x_channel = x[:, :, i]  # Get the current channel of x
	        h_channel = h[:, :, i]  # Get the current channel of h

	        h_pad = np.pad(h_channel, ((p0 // 2, p0 // 2), (p1 // 2, p1 // 2)))
	        Fh = self.to_fourier_domain(h_pad)
	        Fx = self.to_fourier_domain(x_channel)
	        result[:, :, i] = self.to_image_domain(Fx * np.conj(Fh))

	    return result

