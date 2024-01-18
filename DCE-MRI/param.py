'''
	Calculating y = MFSx
'''

import numpy as np
from operators import SignalProcessor 

class Parameters:
	def __init__(self):
		self.signal_processor = SignalProcessor()

	def get_log_spectrum(self, fft_data):

		return 20 * np.log(np.abs(fft_data) + 1e-16)

	def Multi_CoilData(self, x, mask):

		spectrum_image = self.signal_processor.to_fourier_domain(x)
		masked_spectrum = spectrum_image * mask
		zero_filled_ifft = self.signal_processor.to_image_domain(masked_spectrum)

		return spectrum_image, masked_spectrum, zero_filled_ifft
