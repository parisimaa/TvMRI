'''

'''

# --------------------------------------
# Import necessary packages

import numpy as np
from matplotlib import pyplot as plt
from operators import SignalProcessor
from DataLoader import MRIDataLoader
from mask import MaskGenerator
from param import Parameters
from tv import TV_MRI_Sub, TV_MRI_Sub_Ref
from skimage.metrics import peak_signal_noise_ratio as PSNR

# --------------------------------------
# Data Loader
path1 = "data1.mat"
path2 = "data2.mat"

data_loader1 = MRIDataLoader(path1)
data_loader1.load_data()
data_loader1.normalize_single_coil()
data_loader1.normalize_multi_coil()

s1 = data_loader1.load_data()   # Sensitivity Map Img1
x1 = data_loader1.reshape_normalized_data()   # Normalized combined Img1 (256,256,1)

data_loader2 = MRIDataLoader(path2)
data_loader2.load_data()
data_loader2.normalize_single_coil()
data_loader2.normalize_multi_coil()
s2 = data_loader2.load_data()   # Sensitivity Map Img2
x2 = data_loader2.reshape_normalized_data()   # Normalized combined Img2 (256,256,1)

# Print the image information
info2 = data_loader2.get_info()
print("Data Info:")
for key, value in info2.items():
    print(f"{key}: {value}")

# --------------------------------------
# Mask Generation
np.random.seed(30)  
mask_generator = MaskGenerator(x1.shape)  # x1 and x2 have same shapes
rmask1_4x = mask_generator.random_mask(center_percentage=8, accl = 4)   # for img1
rmask1_2x = mask_generator.random_mask(center_percentage=16, accl = 2)   # for img1
rmask2_4x = mask_generator.random_mask(center_percentage=8, accl = 4)   # for img2

# Generate y1 and y2 (MFSx)
pc = Parameters()

# Apply Sensitivity map on combined_2DImg
Sx1 = s1 * x1 
Sx2 = s2 * x2

# Masking in Fourier Domain
spectrum_image1, y1, zero_filled_ifft1 = pc.Multi_CoilData(Sx1, rmask1_4x)
spectrum_image2, y2, zero_filled_ifft2 = pc.Multi_CoilData(Sx2, rmask2_4x)
spectrum_image3, y3, zero_filled_ifft3 = pc.Multi_CoilData(Sx1, rmask1_2x) # Img1 2x acceleration

# --------------------------------------
# Running TV MRI - Magnitude Subtraction CS

rec_x1, rec_x2, J1, J2 = TV_MRI_Sub_Ref(y3, y2, rmask1_2x, rmask2_4x, s1, s2, mu = 1e-4, lamb = 1e-4, maxiter=1000, tol=1e-8)

fig = plt.figure(figsize=(4,4))

plt.imshow(np.abs(rec_x1), cmap='gray')
plt.title('Recovered Image 1, Independent CS')
#plt.title('Recovered Image 1, Magnitude Subtraction CS')

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(4,4))

plt.imshow(np.abs(rec_x2), cmap='gray')
plt.title('Recovered Image 2, Independent CS')
#plt.title('Recovered Image 2, Magnitude Subtraction CS')

plt.tight_layout()
plt.show()

# --------------------------------------
print("Refrence based reconstruction")
print(f"PSNR recovered Img1: {PSNR(np.abs(x1), np.abs(rec_x1),data_range=np.max(np.abs(x1)))}")
print(f"PSNR recovered Img2: {PSNR(np.abs(x2), np.abs(rec_x2),data_range=np.max(np.abs(x2)))}")
# --------------------------------------

rect_x1, rect_x2, J11, J22 = TV_MRI_Sub(y3, y2, rmask1_2x, rmask2_4x, s1, s2, mu = 1e-4, lamb = 1e-4, maxiter=200, tol=1e-8, coord = 5)

print("Regular MS reconstruction")
print(f"PSNR recovered Img1: {PSNR(np.abs(x1), np.abs(rect_x1),data_range=np.max(np.abs(x1)))}")
print(f"PSNR recovered Img2: {PSNR(np.abs(x2), np.abs(rect_x2),data_range=np.max(np.abs(x2)))}")
# --------------------------------------
# Define values for mu and lamb
# mu_values = [0, 1e-6, 1e-5, 1e-4, 1e-3]
# lamb_values = [1e-5, 1e-4, 1e-3]

# # Initialize arrays to store PSNR values
# psnr_x_t1 = np.zeros((len(lamb_values), len(mu_values)))
# psnr_x_t2 = np.zeros((len(lamb_values), len(mu_values)))

# # Iterate over lamb and mu values
# for i, lamb in enumerate(lamb_values):
#     for j, mu in enumerate(mu_values):

#         rec_x1, rec_x2, _, _ = TV_MRI_Sub(y1, y2, rmask1_4x, rmask2_4x, s1, s2, mu=mu, lamb=lamb, maxiter=50, coord=5)

#         # Calculate PSNR for x_t1 and x_t2
#         psnr_x_t1[i, j] = PSNR(np.abs(x1), np.abs(rec_x1), data_range=np.max(np.abs(x1)))
#         psnr_x_t2[i, j] = PSNR(np.abs(x2), np.abs(rec_x2), data_range=np.max(np.abs(x2)))

# # Create subplots
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# for i in range(len(lamb_values)):
#     plt.plot(mu_values, psnr_x_t1[i, :], label=f'lamb={lamb_values[i]}')
# plt.xlabel('mu')
# plt.ylabel('PSNR')
# plt.title('PSNR vs. mu for x_t1')
# plt.legend()

# plt.subplot(1, 2, 2)
# for i in range(len(lamb_values)):
#     plt.plot(mu_values, psnr_x_t2[i, :], label=f'lamb={lamb_values[i]}')
# plt.xlabel('mu')
# plt.ylabel('PSNR')
# plt.title('PSNR vs. mu for x_t2')
# plt.legend()

# plt.tight_layout()
# plt.show()

















