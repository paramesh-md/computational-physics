import numpy as np 
import math
from matplotlib import pyplot as plt, colors
import time

import scipy
from scipy.integrate import quad



def forward_Kn(alpha, z, n):
    """"Function calculates the value of integral using forward recursion. It calculates values at n+1 based on n. The inputs are alpha, z and n(the number of increments).
    Returns K_(n+1), A list of Kn values at each iteration and the number of iterations, n"""""

    K_for = []
    n_points = []
    for i in range(n):
        n_points.append(i)
        if i == 0:
            K = (1/alpha) * (math.log(1 + alpha/z))             ## Calculate K0
            K_for.append(K)
        
            
        else:
            Kn = 1/(alpha*(i+1)) - (z/alpha)*K                  # Iterate for each n+1 value based on n
            K_for.append(Kn)  

            K = Kn                                              # Update value of K for next iteration
                                              
            #print(Kn)
    return [Kn, K_for, n_points]


def downward_Kn(alpha, z, n):
    """"Function calculates the value of integral using backward recursion. It calculates values at n based on n+1. The inputs are alpha, z and n(the number of increments).
    Returns K_(n), A list of Kn values at each iteration and the number of iterations, n"""""

    epsilon = 10E-12
    K = 0
    n_points = []
    K_down = []

    m = math.log(epsilon)/math.log(alpha/z)
    
    for val, i in enumerate(range(n, -1, -1)):                          
        n_points.append(i)
        if val == 0: 
            for j in range(int(m)):
                K1 = ((-alpha)**j  / z**(j+1) ) * (1 / (i + j + 2))         # Sum over m          
                K = K + K1                                                  ## Calculate Kn

            
            K_down.append(K)


        else:
            
            Kn = (1/ z*(i + 1)) - (alpha/z)*K                               # Update value of K for next iteration
            K = Kn                                                          # Fing K0
            K_down.append(Kn)
            

    return [Kn, K_down, n_points]
            


def integrand(x, alpha, z, n):
    value = x**n/(z + alpha*x)
    return value

def main():
    alpha = 1
    z = 1000
    n = 1000                                                                # Number of Iterations

    Recursion_type = ["Upward Recursion", "Downward Recursion"]

    Exact_K = []   

    if (alpha/z) >= 1.0:                                                    # Check for ratio of alpha/z
        result = forward_Kn(alpha, z, n)
        label = Recursion_type[0]

        for k in range(n):
            I = quad(integrand, 0, 1, args=(alpha, z, k))                  # Compute exact value of integral using scipy
            Exact_K.append(I[0])

    else:
        result = downward_Kn(alpha, z, n)
        label = Recursion_type[1]
        for k in range(n+1):
            I = quad(integrand, 0, 1, args=(alpha, z, k))
            Exact_K.append(I[0])
        
        print(Exact_K[0], result[1][-1])
        

            

    print(len(Exact_K), len(result[2]), len(result[1]))
    
    plt.plot(result[2], result[1], label = label)
    plt.plot(result[2], Exact_K, label = "Exact Solution")
    plt.xlabel('n = Number of iterations')
    plt.ylabel('Kn')
    plt.title(label + ' with \u03B1/z = %.2f' % (alpha/z) +', percentage error = %.3f' % ((Exact_K[0] - result[1][-1])/Exact_K[0] *100))
    if label == Recursion_type[1]:
        plt.xlim(max(result[2]),min(result[2]))
    plt.legend()
    plt.show()


                            

    #print(I)



if __name__ == "__main__":
    main()




