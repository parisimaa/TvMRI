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

<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\x^{k+1}%20=%20\prox_{\tau%20g}%20\left(x^k%20-%20\tau%20D^\top%20z^k\right)" />
</p>
<p align="center">
  <img src="https://latex.codecogs.com/svg.latex?\color{white}\z^{k+1}%20=%20\prox_{\sigma%20f^\ast}%20\left(z^k%20+%20\sigma%20D(2x^{k+1}%20-%20x^k)\right)" />
</p>









## 2. Spatio-Temporal TV (Dynamic CS)
## 3. Magnitude Subtraction CS
## 4. Reference based Magnitude Subtraction CS

### Contact
Parisima Abdali: [pa2297@nyu.edu](mailto:pa2297@nyu.edu)
