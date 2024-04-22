import sys
sys.path.insert(0, "../") # Adds the module to path
import deeptrack as dt

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the optics
IMAGE_SIZE = 256
NA = 1.3
MAGNIFICATION = 1
WAVELENGTH = 633e-9
RESOLUTION = 1.14e-7
OPTICS_CASE = "iscat"

# Define the parameters of the particles
RADIUS_RANGE = (25e-9, 200e-9)
REFRACTIVE_INDEX = (1.35, 1.6)
N_PARTICLES = 50
Z_RANGE = (-10, 10)

# Define the parameters of the noise - Need to be tuned
NOISE = 1e-5

def main():

    if OPTICS_CASE == "brightfield":
        optics = dt.Brightfield(
            NA=NA,
            magnification=MAGNIFICATION,
            wavelength=WAVELENGTH,
            resolution=RESOLUTION,
            output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
            return_field=True
        )
    elif OPTICS_CASE == "darkfield":
        optics = dt.Darkfield(
            NA=NA,
            magnification=MAGNIFICATION,
            wavelength=WAVELENGTH,
            resolution=RESOLUTION,
            output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
            illumination_angle = np.pi
        )
    elif OPTICS_CASE == "iscat":
        optics = dt.ISCAT(
            NA=NA,
            magnification=MAGNIFICATION,
            wavelength=WAVELENGTH,
            resolution=RESOLUTION,
            output_region = (0, 0, IMAGE_SIZE, IMAGE_SIZE),
            illumination_angle = np.pi
        )

    #Define the particles
    particles = dt.MieSphere(
        radius=lambda: np.random.uniform(*RADIUS_RANGE),
        refractive_index=lambda: np.random.uniform(*REFRACTIVE_INDEX),
        position = lambda: np.random.uniform(6, IMAGE_SIZE-6, 2),
        z=lambda: np.random.uniform(*Z_RANGE),
        L=100,
        ) ^ N_PARTICLES
    
    #Define the optics and particles.
    training_data = optics(particles) 

    #Gaussian noise
    if NOISE > 0:
        training_data = training_data >> dt.Gaussian(sigma=lambda: np.random.rand() * NOISE)
    
    if OPTICS_CASE == "brightfield":
        training_data = (training_data >> dt.Real()) & (training_data >> dt.Imag())
        training_data = training_data >> dt.Merge(lambda: lambda x: np.concatenate( [np.array(_x) for _x in x], axis=-1 ))

    #Generate the data
    frame = training_data.update().resolve()
    frame = np.array(frame, dtype = np.float32)

    #Save the data
    np.save(f"..data/{OPTICS_CASE}_data.npy", frame)

if __name__ == "__main__":
    main()