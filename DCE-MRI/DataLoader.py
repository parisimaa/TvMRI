'''
   DATA LOADER
   Single coil and Multi - coil MRI data

'''

import numpy as np
import scipy.io as sio

class MRIDataLoader:

	def __init__(self, data_path):

		self.data_path = data_path
		self.data = None
		self.coil_data = None
		self.sensitivity_map = None
		self.data_combined = None
		self.normalized_data = None
		self.normalized_coil_data = None

	def load_data(self):

		img = sio.loadmat(self.data_path)
		self.data = img['Img2D_combined']
		self.coil_data = img['Img2D']
		self.sensitivity_map = img['b11']

		return self.sensitivity_map

    # Combine slices manually using sensitivity map 
    # (use this in case the .mat didn't have Img2D_combined info)

	def preprocess_data(self):

		a = np.sum(self.sensitivity_map * np.conjugate(self.sensitivity_map), axis=2, keepdims=True)
		sensitivity_map_normalized = self.sensitivity_map / np.sqrt(np.abs(a))
		self.data_combined = np.sum(np.conjugate(sensitivity_map_normalized) * self.coil_data, axis=2)

	def normalize_single_coil(self):

		# Normalize single-coil data (256, 256)

		data_min = np.min(np.abs(self.data))
		data_max = np.max(np.abs(self.data))
		self.normalized_data = (self.data - data_min) / (data_max - data_min)

	def normalize_multi_coil(self):

		# Normalize each coil separately using broadcasting (256, 256, 18)

		data_min = np.min(np.abs(self.coil_data), axis=(0, 1), keepdims=True)
		data_max = np.max(np.abs(self.coil_data), axis=(0, 1), keepdims=True)
		self.normalized_coil_data = (self.coil_data - data_min) / (data_max - data_min)

	def reshape_normalized_data(self):
		# Reshape the normalized data to (256, 256, 1)
		return self.normalized_data[:, :, np.newaxis]

	def get_info(self):
		return {
			'Data Shape': self.data.shape if self.data is not None else None,
			'Coil Data Shape': self.coil_data.shape if self.coil_data is not None else None,
			'Sensitivity Map Shape': self.sensitivity_map.shape if self.sensitivity_map is not None else None,
			'Normalized Data Shape': self.normalized_data.shape if self.normalized_data is not None else None,
			'Normalized Coil Data Shape': self.normalized_coil_data.shape if self.normalized_coil_data is not None else None,
		}








