'''
	TV - MRI Algorithms 
'''

import numpy as np
from operators import SignalProcessor  # Import SignalProcessor from operators.py

SignalP = SignalProcessor()

dh = np.array([[1,-1],[0, 0]]) # horizontal gradient filter
dv = np.array([[1, 0],[-1,0]]) # vertical gradient filter

dh = dh.reshape(2, 2, 1)
dv = dv.reshape(2, 2, 1)

Dh = lambda x: SignalP.conv2d_fft(x, dh)
Dv = lambda x: SignalP.conv2d_fft(x, dv)

DhT = lambda x: SignalP.conv2dT_fft(x, dh)
DvT = lambda x: SignalP.conv2dT_fft(x, dv)



def TV_MRI_Sub(y1, y2, M1, M2, sens1, sens2, mu =1, lamb = 2, maxiter=50, coord = 1, tol=1e-4 ):

  # define the soft-thresholding function for complex numbers
  soft_thresh = lambda v, t: np.maximum(np.abs(v)-t, 0.)*np.exp(1j*np.angle(v))

# -----------------------------
  # set step-sizes at maximum: τσL² < 1
  # note: PDS seems sensitive to these (given finite iterations at least...)
  L = np.sqrt(12) # Spectral norm of D
  tao = 0.99 / L
  sigma = 0.99 / L
# -----------------------------
  # Proximal Gradient Descent on x (primal)
  prox_D = lambda v, b: soft_thresh(v - b, tao*mu) + b
  # Proximal Gradient Ascent on z (dual)
  def prox_A(v):
    magnitude = np.abs(v)

    scaling_factor = np.maximum(1, magnitude / lamb)

    clipped_image = v / scaling_factor

    return clipped_image

  # Sensitivity encoding operator
  Sh = lambda s,v: np.sum(np.conjugate(s)*v, axis = 2, keepdims=True)
# -----------------------------
  # initilize iteration variables
  zh_1 = np.zeros((256, 256, 1))
  zv_1 = np.zeros((256, 256, 1))

  x_1 = SignalP.to_image_domain(np.zeros((256, 256, 1)))

  zh_2 = np.zeros((256, 256, 1))
  zv_2 = np.zeros((256, 256, 1))

  x_2 = SignalP.to_image_domain(np.zeros((256, 256, 1)))


  # For computing convergence
  J1 = np.zeros(maxiter)
  J2 = np.zeros(maxiter)

  #for k in range(maxiter):
  # Iterations
  k = 0
  while k < maxiter :

      # Coordinate Descent
      for i in range(coord):
        # Update x1 - ProxD
          x1_old = x_1
          phase1 = np.angle(x_1)
          v1 = SignalP.to_image_domain((M1*SignalP.to_fourier_domain(sens1*x_1))-y1)
          Sens1 = Sh(sens1,v1)
          x_1 = prox_D(x_1 - tao*(Sens1 + (DhT(zh_1) + DvT(zv_1))), np.abs(x_2)*np.exp(1j*phase1))

        # Update x2 - ProxD
          x2_old = x_2
          phase2 = np.angle(x_2)
          v2 = SignalP.to_image_domain((M2*SignalP.to_fourier_domain(sens2*x_2))-y2)
          Sens2 = Sh(sens2,v2)
          x_2 = prox_D(x_2 - tao*(Sens2 + (DhT(zh_2) + DvT(zv_2))), np.abs(x_1)*np.exp(1j*phase2))

    # Update z1 - ProxA
      zh_1 = prox_A(zh_1 + sigma*(Dh(2*x_1 - x1_old)))
      zv_1 = prox_A(zv_1 + sigma*(Dv(2*x_1 - x1_old)))
    # Update z2 - ProxA
      zh_2 = prox_A(zh_2 + sigma*(Dh(2*x_2 - x2_old)))
      zv_2 = prox_A(zv_2 + sigma*(Dv(2*x_2 - x2_old)))

    # compute the convergence (residual)
      J1[k] = np.abs(x_1 - x1_old).max()
      J2[k] = np.abs(x_2 - x2_old).max()

      #print(f"{k} | J1={J1[k]} | J2={J2[k]}")
      if J1[k] < tol and J2[k] < tol and k >2:
          break
      k=k+1

  return x_1, x_2, J1, J2


def TV_MRI_Sub_Ref(y1, y2, M1, M2, sens1, sens2, mu =1, lamb = 2, maxiter=50, tol=1e-6):

  # define the soft-thresholding function for complex numbers
  soft_thresh = lambda v, t: np.maximum(np.abs(v)-t, 0.)*np.exp(1j*np.angle(v))

# -----------------------------
  # set step-sizes at maximum: τσL² < 1
  # note: PDS seems sensitive to these (given finite iterations at least...)
  L = np.sqrt(12) # Spectral norm of D
  tao = 0.99 / L
  sigma = 0.99 / L
# -----------------------------
  # Proximal Gradient Descent on x (primal)
  prox_D = lambda v, b: soft_thresh(v - b, tao*mu) + b
  # Proximal Gradient Ascent on z (dual)
  def prox_A(v):
    magnitude = np.abs(v)

    scaling_factor = np.maximum(1, magnitude / lamb)

    clipped_image = v / scaling_factor

    return clipped_image

  # Sensitivity encoding operator
  Sh = lambda s,v: np.sum(np.conjugate(s)*v, axis = 2, keepdims=True)
# -----------------------------
  # initilize iteration variables
  zh_1 = np.zeros((256, 256, 1))
  zv_1 = np.zeros((256, 256, 1))

  x_1 = SignalP.to_image_domain(np.zeros((256, 256, 1)))

  zh_2 = np.zeros((256, 256, 1))
  zv_2 = np.zeros((256, 256, 1))

  x_2 = SignalP.to_image_domain(np.zeros((256, 256, 1)))


  # For computing convergence
  J1 = np.zeros(maxiter)
  J2 = np.zeros(maxiter)

  #for k in range(maxiter):
  # Iterations
  k = 0
  while k < maxiter :

      # Coordinate Descent
      
    # Update x1 - ProxD
      x1_old = x_1
      phase1 = np.angle(x_1)
      v1 = SignalP.to_image_domain((M1*SignalP.to_fourier_domain(sens1*x_1))-y1)
      Sens1 = Sh(sens1,v1)
      x_1 = prox_D(x_1 - tao*(Sens1 + (DhT(zh_1) + DvT(zv_1))), 0)

    # Update z1 - ProxA
      zh_1 = prox_A(zh_1 + sigma*(Dh(2*x_1 - x1_old)))
      zv_1 = prox_A(zv_1 + sigma*(Dv(2*x_1 - x1_old)))

    # compute the convergence (residual)
      J1[k] = np.abs(x_1 - x1_old).max()

      #print(f"{k} | J1={J1[k]} | J2={J2[k]}")
      if J1[k] < tol and k >2:
          break
      k=k+1

  k = 0
  while k < maxiter :
      # Update x2 - ProxD
      x2_old = x_2
      phase2 = np.angle(x_2)
      v2 = SignalP.to_image_domain((M2*SignalP.to_fourier_domain(sens2*x_2))-y2)
      Sens2 = Sh(sens2,v2)
      x_2 = prox_D(x_2 - tao*(Sens2 + (DhT(zh_2) + DvT(zv_2))), np.abs(x_1)*np.exp(1j*phase2))

      # Update z2 - ProxA
      zh_2 = prox_A(zh_2 + sigma*(Dh(2*x_2 - x2_old)))
      zv_2 = prox_A(zv_2 + sigma*(Dv(2*x_2 - x2_old)))

      J2[k] = np.abs(x_2 - x2_old).max()
      
      #print(f"{k} | J1={J1[k]} | J2={J2[k]}")
      if J2[k] < tol and k >2:
          break
      k=k+1

  return x_1, x_2, J1, J2
