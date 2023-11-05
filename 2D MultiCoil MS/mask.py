'''
	MASKS / UNDERSAMPLING
'''

# Random mask
'''

'''
import numpy as np

class MaskGenerator:

	def __init__(self, img_shape):
		self.img_shape = img_shape

	def random_mask(self, center_percentage=8, accl = 4):

		mask = np.zeros(self.img_shape)

		center_columns = int(self.img_shape[1] * center_percentage / 100)
		left_padding = (self.img_shape[1] - center_columns) // 2
		right_padding = self.img_shape[1] - center_columns - left_padding

		# Define the center region
		mask[:, left_padding:left_padding + center_columns] = 1

		# Determine the number of random columns outside the center region
		num_random_cols = int((1 / accl) * self.img_shape[1])
		random_cols = np.random.choice(
		    [i for i in range(self.img_shape[1]) if i < left_padding or i >= left_padding + center_columns],
		    num_random_cols,
		    replace=False
		)

		# Set the random columns to 1 in the mask
		mask[:, random_cols] = 1

		return mask.astype(int)

	

# Work on this part of the code later ........



	# def equidistant_mask(self, center_percentage = 8, accl = 4):

	# 	mask = np.zeros(self.img_shape)
	# 	mask[::accl] = 1.0

	# 	# Set center_percentage% of the center lines to all ones

	# 	center_lines = self.img_shape[0] // (center_percentage)
	# 	start = (self.img_shape[0] - center_lines) // 2
	# 	end = start + center_lines
	# 	mask[start:end] = 1.0
	# 	return np.transpose(mask, (1, 0, 2))




	def equidistant_mask(self, center_percentage=0.08, accl=4):
		mask = np.zeros(self.img_shape)

		center_columns = int(self.img_shape[1] * center_percentage / 100)
		left_padding = (self.img_shape[1] - center_columns) // 2
		right_padding = self.img_shape[1] - center_columns - left_padding

		# Define the center region
		mask[:, left_padding:left_padding + center_columns] = 1

		# Determine the number of random columns outside the center region
		num_random_cols = int((1 / accl) * (self.img_shape[1] - center_columns))
		if num_random_cols > 0:
		    # Calculate equidistant spacing for random columns
		    spacing = (self.img_shape[1] - center_columns) / (num_random_cols + 1)

		    # Calculate the positions of random columns
		    random_cols = [int(left_padding + (i + 1) * spacing) for i in range(num_random_cols)]

		    # Set the random columns to 1 in the mask
		    mask[:, random_cols] = 1

		return mask.astype(int)






