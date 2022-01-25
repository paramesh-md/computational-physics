import numpy as np 
from matplotlib import pyplot as plt, colors
import time
from numba import jit  # This is the new line with numba
        
@jit(nopython=True)    # This is the second new line with numba
def mandelbrot(z, iter):
    z_0 = z
    for n in range(iter):
        if abs(z) >2:
            return n
        
        z = z*z + z_0

    return iter


def mandelbrot_set(xmin, xmax, ymin, ymax, w, h, iter):
    real_x = np.linspace(xmin, xmax, w)
    img_y = np.linspace(ymin, ymax, h)
    data = np.empty((w, h))
    for i in range(w):
        for j in range(h):
            data[i][j] = mandelbrot(real_x[i] + 1j*img_y[j], iter)

    return [real_x, img_y, data]



def mandelbrot_image(xmin,xmax,ymin,ymax,width=3,height=3,maxiter=80,cmap='hot'):
    dpi = 72
    img_width = dpi * width
    img_height = dpi * height
    x,y,z = mandelbrot_set(xmin,xmax,ymin,ymax,img_width,img_height,maxiter)
    
    fig, ax = plt.subplots(figsize=(width, height),dpi=72)
    ticks = np.arange(0,img_width,3*dpi)

    

    x_ticks = xmin + (xmax-xmin)*ticks/img_width
    #plt.xticks(ticks, xmax-xmin)
    y_ticks = ymin + (ymax-ymin)*ticks/img_width
    #plt.yticks(ticks, y_ticks)
    print(x_ticks, y_ticks)
    norm = colors.PowerNorm(0.3)
    ax.imshow(z.T,cmap=cmap,origin='lower',norm=norm, extent = [xmin,xmax,ymin,ymax] )
    plt.show()

t0 = time.time()
result = mandelbrot_set(-2.0, 0.5, -1.25, 1.25, 200, 200, 200)
t1 = time.time()
print("Python", t1-t0)
mandelbrot_image(-2.0,0.5,-1.25,1.25,maxiter=80,cmap='gnuplot2')
