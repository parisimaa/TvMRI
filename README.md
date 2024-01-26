# TvMRI
TV - MRI Reconstruction Compressed Sensing \
Download the dataset from: https://drive.google.com/file/d/0B4nLrDuviSiWajFDV1Frc3cxR0k/view?usp=sharing

## Compressed Senssing: 
## Parallel Imaging: 

## 1. Spatial TV (Independent CS)
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\min_{x}%20\quad%20\frac{1}{2}\left\|y-MFx\right\|_{2}^{2}+\lambda\left\|Dx\right\|_{1}" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}D%20=%20\begin{bmatrix}%20D_h\\%20D_v%20\end{bmatrix}%20\quad|%20\quad%20D_{hx}%20=%20\begin{bmatrix}%201\\%20-1%20\end{bmatrix}%20\ast%20x%20\quad|%20\quad%20D_{vx}%20=%20\begin{bmatrix}%201%20&%20-1%20\end{bmatrix}%20\ast%20x" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\min_{x}%20f(Dx)%20+%20g(x)" />
</p>

Iteration updates:
 <p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white} x^{k+1}%20=%20\text{prox}_{\tau%20g}%20\left(x^k%20-%20\tau%20D^\top%20z^k\right)" />
</p>
<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white} z^{k+1}%20=%20\text{prox}_{\sigma%20f^*}%20\left(z^k%20+%20\sigma%20D(2x^{k+1}%20-%20x^k)\right)" />
</p>


## 2. Spatio-Temporal TV (Dynamic CS)

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\min_{x}%20\quad%20\frac{1}{2}\left\|y-MFSx\right\|_{2}^{2}+\lambda_s\left\|D_sx\right\|_{1}%20+%20\lambda_t\left\|D_tx\right\|_{1}" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}(D_s%20x)_{ijk}%20=%20\begin{bmatrix}x_{ijk}%20-%20x_{(i-1)jk}%20\\%20x_{ijk}%20-%20x_{i(j-1)k}\end{bmatrix}" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}(D_t%20x)_{ijk}%20=%20x_{ijk}%20-%20x_{ij(k-1)}" />
</p>

Iteration updates:
<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}x^{k+1}%20=%20x^k%20-%20\tau%20(S^HF^H(MFSx^k%20-%20y)%20+%20D_s^\top%20z_s^k%20+%20D_t^\top%20z_t^k)" />

<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}z_s^{k+1}%20=%20\mathcal{S}_{\lambda_s}%20\left(z_s^k%20+%20\sigma%20D_s%20(2x^{k+1}-x^{k})\right)" />
</p>
<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}z_t^{k+1}%20=%20\mathcal{S}_{\lambda_t}%20\left(z_t^k%20+%20\sigma%20D_t%20(2x^{k+1}-x^{k})\right)" />
</p>

## 3. Magnitude Subtraction CS

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\min_{x_1%20,%20x_2}%20\quad%20\frac{1}{2}\left\|%20y_1%20-%20M_1%20FS_1x_1%20\right\|_{2}^{2}%20+%20\frac{1}{2}\left\|%20y_2%20-%20M_2%20FS_2x_2\right\|_{2}^{2}%20+%20\lambda%20TV\left(x_1\right)%20+%20\lambda%20TV\left(x_2\right)%20+%20\mu%20\left\|%20\left%20|x_2%20\right|%20-%20\left%20|x_1%20\right|\right\|_{1}" />
</p>

Iteration updates:
<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}x_1^{n+1}%20=%20\mathcal{S}_{\tau%20\mu}%20\left(\left(x_1^n%20-%20\tau%20\left(S_1^HF^H(M_1FS_1x_1^n-y_1)+D^Tz_1^n\right)\right)-\left%20|x_2^n%20\right|%20\left(e^{i\phi_1}\right)\right)%20+%20\left%20|x_2^n%20\right|%20\left(e^{i\phi_1}\right)" />
</p>
<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}x_2^{n+1}%20=%20\mathcal{S}_{\tau%20\mu}%20\left(\left(x_2^n%20-%20\tau%20\left(S_2^HF^H(M_2FS_2x_2^n-y_2)+D^Tz_2^n\right)\right)-\left%20|x_1^n%20\right|%20\left(e^{i\phi_2}\right)\right)%20+%20\left%20|x_1^n%20\right|%20\left(e^{i\phi_2}\right)" />
</p>

<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}z_1^{n+1}%20=%20\text{clip}_\lambda%20\left(z_1^n%20+%20\sigma%20D\left(2x_1^{n+1}%20-%20x_1^n\right)\right)" />
</p>
<p >
  <img src="https://latex.codecogs.com/svg.latex?\color{white}z_2^{n+1}%20=%20\text{clip}_\lambda%20\left(z_2^n%20+%20\sigma%20D\left(2x_2^{n+1}%20-%20x_2^n\right)\right)" />
</p>


## 4. Reference based Magnitude Subtraction CS
his method involves reconstructing the first frame at a lower acceleration rate
independently and utilizing its magnitude as a reference for the reconstruction of subsequent images.
By adopting this strategy, we can improve the Peak Signal-to-Noise Ratio (PSNR). When applying
this approach to all frames within a dataset, we consistently observe that the PSNR exceeds that of
independent reconstructions throughout the sequence.

### Contact
Parisima Abdali: [pa2297@nyu.edu](mailto:pa2297@nyu.edu)
