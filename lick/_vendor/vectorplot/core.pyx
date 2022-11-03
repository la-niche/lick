"""
Algorithm based on "Imaging Vecotr Fields Using Line Integral Convolution"
                   by Brian Cabral and Leith Leedom
"""
cimport cython
cimport numpy as np
from libc.math cimport signbit


cdef fused real:
    np.float64_t
    np.float32_t

@cython.cdivision
cdef inline void _advance(real* vx,
        int* x, int* y, real* fx, int w, int h):
    """Move to the next pixel in the vector direction.

    This function updates x, y, fx, and fy in place.

    Parameters
    ----------
    vx : real
      Vector x component.
    x : int
      Pixel x index. Updated in place.
    y : int
      Pixel y index. Updated in place.
    fx : float
      Position along x in the pixel unit square. Updated in place.
    w : int
      Number of pixels along x.
    h : int
      Number of pixels along y.
    """

    cdef real tx
    cdef real ty

    # Think of tx (ty) as the time it takes to reach the next pixel
    # along x (y).

    if vx[0]==0 and vx[1]==0:
        return

    tx = (signbit(-vx[0])-fx[0])/vx[0]
    ty = (signbit(-vx[1])-fx[1])/vx[1]

    if tx<ty:    # We reached the next pixel along x first.
        if vx[0]>=0:
            x[0]+=1
            fx[0]=0
        else:
            x[0]-=1
            fx[0]=1
        fx[1]+=tx*vx[1]
    else:        # We reached the next pixel along y first.
        if vx[1]>=0:
            y[0]+=1
            fx[1]=0
        else:
            y[0]-=1
            fx[1]=1
        fx[0]+=ty*vx[0]

    x[0] = max(0, min(w-1, x[0]))
    y[0] = max(0, min(h-1, y[0]))

@cython.boundscheck(False)
@cython.wraparound(False)
def line_integral_convolution(
        np.ndarray[real, ndim=2] u,
        np.ndarray[real, ndim=2] v,
        np.ndarray[real, ndim=2] texture,
        np.ndarray[real, ndim=1] kernel,
        np.ndarray[real, ndim=2] out,
        int polarization=0):
    """Return an image of the texture array blurred along the local
    vector field orientation.
    Parameters
    ----------
    u : array (ny, nx)
      Vector field x component.
    v : array (ny, nx)
      Vector field y component.
    texture : array (ny,nx)
      The texture image that will be distorted by the vector field.
      Usually, a white noise image is recommended to display the
      fine structure of the vector field.
    kernel : 1D array
      The convolution kernel: an array weighting the texture along
      the stream line. For static images, a box kernel (equal to one)
      of length max(nx,ny)/10 is appropriate. The kernel should be
      symmetric.

    out : 2D array
      The result array. Initial state should be all zeros
    polarization : int (0 or 1)
      If 1, treat the vector field as a polarization (so that the
      vectors have no distinction between forward and backward).

    Returns
    -------
    out : array(ny,nx)
      An image of the texture convoluted along the vector field
      streamlines.

    """

    cdef int i,j,k,x,y
    cdef int kernellen
    cdef real last_ui, last_vi
    cdef int pol = polarization

    cdef real ui[2]
    cdef real fx[2]
    cdef real[:, :] u_v = u
    cdef real[:, :] v_v = v
    cdef real[:, :] texture_v = texture
    cdef real[:, :] out_v = out

    ny = u.shape[0]
    nx = u.shape[1]

    kernellen = kernel.shape[0]


    for i in range(ny):
        for j in range(nx):
            x = j
            y = i
            fx[0] = 0.5
            fx[1] = 0.5
            last_ui = 0
            last_vi = 0

            k = kernellen//2
            out_v[i,j] += kernel[k]*texture_v[y,x]

            while k<kernellen-1:
                ui[0] = u_v[y,x]
                ui[1] = v_v[y,x]
                if pol and (ui[0]*last_ui+ui[1]*last_vi)<0:
                    ui[0] = -ui[0]
                    ui[1] = -ui[1]
                last_ui = ui[0]
                last_vi = ui[1]
                _advance(ui,
                        &x, &y, fx, nx, ny)
                k+=1
                out_v[i,j] += kernel[k]*texture_v[y,x]

            x = j
            y = i
            fx[0] = 0.5
            fx[1] = 0.5
            last_ui = 0
            last_vi = 0

            k = kernellen//2

            while k>0:
                ui[0] = -u_v[y,x]
                ui[1] = -v_v[y,x]
                if pol and (ui[0]*last_ui+ui[1]*last_vi)<0:
                    ui[0] = -ui[0]
                    ui[1] = -ui[1]
                last_ui = ui[0]
                last_vi = ui[1]
                _advance(ui,
                        &x, &y, fx, nx, ny)
                k-=1
                out_v[i,j] += kernel[k]*texture_v[y,x]
