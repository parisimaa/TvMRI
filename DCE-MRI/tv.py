'''
	TV - MRI Algorithms 
'''

from re import T
import numpy as np
from operators import SignalProcessor  # Import SignalProcessor from operators.py
import scipy.fft as fft

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
  zh_1 = np.zeros(M1.shape)
  zv_1 = np.zeros(M1.shape)

  x_1 = SignalP.to_image_domain(np.zeros(M1.shape))

  zh_2 = np.zeros(M2.shape)
  zv_2 = np.zeros(M2.shape)

  x_2 = SignalP.to_image_domain(np.zeros(M2.shape))


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
# ------------------------------------------------------------------------------------
def TV_MRI(y1, M1, sens1, lamb = 2, maxiter=50, tol=1e-6):

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
  prox_D = lambda v: soft_thresh(v, 0)
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
  zh_1 = np.zeros(M1.shape)
  zv_1 = np.zeros(M1.shape)

  x_1 = SignalP.to_image_domain(np.zeros(M1.shape))

  # For computing convergence
  J1 = np.zeros(maxiter)

  #for k in range(maxiter):
  # Iterations
  k = 0
  while k < maxiter :
      
    # Update x1 - ProxD
      x1_old = x_1
      phase1 = np.angle(x_1)
      v1 = SignalP.to_image_domain((M1*SignalP.to_fourier_domain(sens1*x_1))-y1)
      Sens1 = Sh(sens1,v1)
      x_1 = prox_D(x_1 - tao*(Sens1 + (DhT(zh_1) + DvT(zv_1))))

    # Update z1 - ProxA
      zh_1 = prox_A(zh_1 + sigma*(Dh(2*x_1 - x1_old)))
      zv_1 = prox_A(zv_1 + sigma*(Dv(2*x_1 - x1_old)))

    # compute the convergence (residual)
      J1[k] = np.abs(x_1 - x1_old).max()

      #print(f"{k} | J1={J1[k]} | J2={J2[k]}")
      if J1[k] < tol and k > 2:
          break
      k=k+1

  return x_1, J1
# ------------------------------------------------------------------------------------
def TV_MRI_Sub_Ref(Ref, y2, M2, sens2, mu =1, lamb = 2, maxiter=50, tol=1e-6):

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
  prox_D = lambda v,b: soft_thresh(v - b, tao*mu) + b
  # Proximal Gradient Ascent on z (dual)
  def prox_A(v):
    magnitude = np.abs(v)

    scaling_factor = np.maximum(1, magnitude / lamb)

    clipped_image = v / scaling_factor

    return clipped_image

  # Sensitivity encoding operator
  Sh = lambda s,v: np.sum(np.conjugate(s)*v, axis = 2, keepdims=True)
# -----------------------------

  zh_2 = np.zeros(M2.shape)
  zv_2 = np.zeros(M2.shape)

  x_2 = SignalP.to_image_domain(np.zeros(M2.shape))

  J2 = np.zeros(maxiter)

  #for k in range(maxiter):
  # Iterations
  k = 0
  while k < maxiter :
      # Update x2 - ProxD
      x2_old = x_2
      phase2 = np.angle(x_2)
      v2 = SignalP.to_image_domain((M2*SignalP.to_fourier_domain(sens2*x_2))-y2)
      Sens2 = Sh(sens2,v2)
      x_2 = prox_D(x_2 - tao*(Sens2 + (DhT(zh_2) + DvT(zv_2))), np.abs(Ref)*np.exp(1j*phase2))

      # Update z2 - ProxA
      zh_2 = prox_A(zh_2 + sigma*(Dh(2*x_2 - x2_old)))
      zv_2 = prox_A(zv_2 + sigma*(Dv(2*x_2 - x2_old)))

      J2[k] = np.abs(x_2 - x2_old).max()
      
      #print(f"{k} | J1={J1[k]} | J2={J2[k]}")
      if J2[k] < tol and k >2:
          break
      k=k+1

  return x_2, J2

# ------------------------------------------------------------------------------------
def TV_MRI_SpatioTemp(y, M, s, slamb = 2, tlamb = 2, maxiter=50, tol=1e-6):

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

  # Proximal Gradient Ascent on z (dual)
  def prox_A(v, l):
    magnitude = np.abs(v)

    scaling_factor = np.maximum(1, magnitude / l)

    clipped_image = v / scaling_factor

    return clipped_image

  #prox_A = lambda v, lamb : np.clip(v, -lamb, lamb)
  # Sensitivity encoding operator
  Sh = lambda s, v: np.sum(np.conjugate(s)*v, axis = 2, keepdims=True)
  # Dt operator
  # Dt = lambda x, t: np.zeros_like(x[:,:,:,t], dtype=np.complex128) if t == 0 else x[:,:,:,t] - x[:,:,:,t-1]
  # DtT = lambda x, t: np.conjugate(Dt(x, t))
  def Dt(u):
      rows, cols, sensitivity_maps, time = u.shape
      d = np.zeros_like(u)
      for i in range(sensitivity_maps):
          d[:, :, i, 1:time] = u[:, :, i, 1:time] - u[:, :, i, 0:time-1]
          d[:, :, i, 0] = u[:, :, i, 0] - u[:, :, i, time-1]
      return d
  def DtT(u):
      rows, cols, sensitivity_maps, time = u.shape
      d = np.zeros_like(u)
      for i in range(sensitivity_maps):
          d[:, :, i, 0:time-1] = u[:, :, i, 0:time-1] - u[:, :, i, 1:time]
          d[:, :, i, time-1] = u[:, :, i, time-1] - u[:, :, i, 0]
      return d

# -----------------------------

  zh_s = np.zeros_like(M, dtype=np.complex128)
  zv_s = np.zeros_like(M, dtype=np.complex128)
  z_t = np.zeros_like(M, dtype=np.complex128)

  x_t = SignalP.to_image_domain(np.zeros(M.shape))
  x_old = np.zeros_like(x_t, dtype=np.complex128)


  J = np.zeros((maxiter, M.shape[-1]))

  #for k in range(maxiter):
  # Iterations
  k = 0
  while k < maxiter :
    #num_frames = M.shape[3]  # Assuming M has shape [height, width, coils, frames]
    for t in range(49):

      # Update x_t1 - ProxD
      x_old[:,:,:,t] = x_t[:,:,:,t]
      b = M[:,:,:,t]* SignalP.to_fourier_domain(s*x_t[:,:,:,t]) + y[:,:,:,t]
      Sens = Sh(s, SignalP.to_image_domain(b))
      x_t[:, :, :, t] = x_t[:,:,:,t] - tao*(Sens + DhT(zh_s[:,:,:,t]) + DvT(zv_s[:,:,:,t]) + DtT(z_t)[:,:,:,t])

      # Update zs - ProxA
      zh_s[:, :, :, t] = prox_A(zh_s[:, :, :, t] + sigma * (Dh(2 * x_t[:, :, :, t] - x_old[:, :, :, t])), slamb)
      zv_s[:, :, :, t] = prox_A(zv_s[:, :, :, t] + sigma * (Dv(2 * x_t[:, :, :, t] - x_old[:, :, :, t])), slamb)
      # c = 2 * x_t - x_old
      z_t[:, :, :, t] = prox_A(z_t[:, :, :, t] + sigma * (Dt(x_t)[:,:,:,t]), tlamb)
      

      # Store convergence history for each frame
    J[k, t] = np.abs(x_t[:, :, :, t] - x_old[:, :, :, t]).max()

    if J[k, :].max() < tol and k > 2:
      break
    
    k += 1
    

  return x_t, J
  # ------------------------------------------------------------------------------------
# def TV_MRI_dynamic(y1, y2, M1, M2, s, slamb = 1, tlamb = 1, maxiter=50, tol=1e-6):

#   # define the soft-thresholding function for complex numbers
#   soft_thresh = lambda v, t: np.maximum(np.abs(v)-t, 0.)*np.exp(1j*np.angle(v))

# # -----------------------------
#   # set step-sizes at maximum: τσL² < 1
#   # note: PDS seems sensitive to these (given finite iterations at least...)
#   L = np.sqrt(12) # Spectral norm of D
#   tao = 0.99 / L
#   sigma = 0.99 / L
# # -----------------------------
#   # Proximal Gradient Descent on x (primal)
#   # Proximal Gradient Ascent on z (dual)
#   def prox_A(v):
#     magnitude = np.abs(v)

#     scaling_factor = np.maximum(1, magnitude / lamb)

#     clipped_image = v / scaling_factor

#     return clipped_image

#   # Sensitivity encoding operator
#   Sh = lambda s,v: np.sum(np.conjugate(s)*v, axis = 2, keepdims=True)
# # -----------------------------
  
#   zh_s1 = np.zeros(M1.shape)
#   zv_s1 = np.zeros(M1.shape)
#   zh_s2 = np.zeros(M2.shape)
#   zv_s2 = np.zeros(M2.shape)

#   z_t1 = np.zeros(M1.shape)
#   z_t2 = np.zeros(M2.shape)

#   x_t1 = SignalP.to_image_domain(np.zeros(M1.shape))
#   x_t2 = SignalP.to_image_domain(np.zeros(M2.shape))

#   J1 = np.zeros(maxiter)

#   #for k in range(maxiter):
#   # Iterations
#   k = 0
#   while k < maxiter :

#       # Update x_t1 - ProxD
#       x_old1 = x_t1
#       b1 = M1 * SignalP.to_fourier_domain(s*x_t1) + y1
#       Sens1 = Sh(s, SignalP.to_image_domain(b1))
#       x_t1 = x_t1 - tao*(Sens1 + DhT(zh_s1) + DvT(zv_s1) + DtT(z_t1))

#       # Update zs - ProxA
#       zh_s1 = prox_A(zh_s1 + sigma * (Dh(2 * x_t1 - x_old1)), slamb)
#       zv_s1 = prox_A(zv_s1 + sigma * (Dv(2 * x_t1 - x_old1)), slamb)
#       c1 = 2 * x_t1 - x_old1
#       z_t1 = prox_A(z_t1 + sigma * (Dt(c)), tlamb)
      
#       J2[k] = np.abs(x_2 - x2_old).max()
      
#       #print(f"{k} | J1={J1[k]} | J2={J2[k]}")
#       if J2[k] < tol and k >2:
#           break
#       k=k+1

#   return x_2, J2

