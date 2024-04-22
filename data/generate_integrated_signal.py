import numpy as np
from scipy.integrate import quad

nm = 1.333 #Refractive index of water
wavelength = 532 #In nanometers
k = 2 * np.pi * nm / wavelength #Wave vector
radius_list = np.arange(1, 301) #Radius range in nanometers
form_factor_list = np.zeros(len(radius_list)) #Empty list to store form factors

def fun(theta, R):
    return (3. / ((2 * k) * np.sin((theta - np.pi / 2) / 2) * R) ** 3 *
            (np.sin(2 * k * np.sin((theta - np.pi / 2) / 2) * R) -
             (2 * k * np.sin((theta - np.pi / 2) / 2) * R) *
             np.cos(2 * k * np.sin((theta - np.pi / 2) / 2) * R))) ** 2

def main():
    for i, R in enumerate(radius_list):
        integrand = lambda theta: fun(theta, R)
        result, _ = quad(integrand, -np.pi / 10, np.pi / 10)
        form_factor_list[i] = result

    np.save('integralSquared.npy', form_factor_list)

if __name__ == '__main__':
    main()
