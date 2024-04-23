import sys
sys.path.insert(0, "../") # Adds the module to path
import deeptrack as dt

import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the optics
IMAGE_SIZE = 512
NA = 1.3
MAGNIFICATION = 1
WAVELENGTH = 633e-9
RESOLUTION = 1.14e-7
OPTICS_CASE = "darkfield" # "brightfield", "darkfield", "iscat"

# Define the parameters of the particles
RADIUS_RANGE = (25e-9, 200e-9)
REFRACTIVE_INDEX = (1.37, 1.6)
N_PARTICLES = 100
Z_RANGE = (-10, 10)

# Define the parameters of the noise - Need to be tuned
NOISE = True
NOISE_DARKFIELD = 5e-5
NOISE_ISCAT = 1e-3
NOISE_BRIGHTFIELD_REAL = 5e-2
NOISE_BRIGHTFIELD_IMAG = 7e-2

# Define the pupil function for noise in holography and ISCAT
def crop(pupil_radius):
    def inner(image):
        x = np.arange(image.shape[0]) - image.shape[0] / 2
        y = np.arange(image.shape[1]) - image.shape[1] / 2
        X, Y = np.meshgrid(x, y)
        image[X ** 2 + Y ** 2 > pupil_radius ** 2] = 0
        return image
    return inner

CROP = dt.Lambda(crop, pupil_radius=lambda: 0.8*IMAGE_SIZE)
HC = dt.HorizontalComa(coefficient=lambda c1: c1, c1=0 + np.random.randn() * 0.5)
VC = dt.VerticalComa(coefficient=lambda c2:c2, c2=0 + np.random.randn() * 0.5)

def get_labels(image):
    array = np.zeros((N_PARTICLES, 5), dtype = np.float32)
    count = 0
    for property in image.properties:
        if "position" in property:
            px = property["position"]
            z = property["z"]
            r = property["radius"]
            n = property["refractive_index"]
            array[count, :] = np.array([px[0], px[1], z, r, n])
            count += 1
    return array

def main():

    if OPTICS_CASE == "brightfield":
        optics = dt.Brightfield(
            NA=NA,
            magnification=MAGNIFICATION,
            wavelength=WAVELENGTH,
            resolution=RESOLUTION,
            output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
            return_field=True,
            pupil= HC >> VC >> CROP
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
            illumination_angle = np.pi,
            pupil = HC >> VC >> CROP
        )

    #Define the particles
    particles = dt.MieSphere(
        radius=lambda: np.random.uniform(*RADIUS_RANGE),
        refractive_index=lambda: np.random.uniform(*REFRACTIVE_INDEX),
        position = lambda: np.random.uniform(6, IMAGE_SIZE-6, 2),
        z=lambda: np.random.uniform(*Z_RANGE),
        L=100) ^ N_PARTICLES
    
    #Define the optics and particles.
    training_data = optics(particles) 

    #Gaussian and poisson noise
    if NOISE == True:
        if OPTICS_CASE == "darkfield":
            training_data = training_data >> dt.Poisson(snr=lambda: 7 + np.random.rand() * 10, background=0) >> dt.Gaussian(sigma=lambda: np.random.rand() * NOISE_DARKFIELD)
        elif OPTICS_CASE == "iscat":
            training_data = training_data >> dt.Poisson(snr=lambda: 7 + np.random.rand() * 10, background=1) >> dt.Gaussian(sigma=lambda: np.random.rand() * NOISE_ISCAT)
        elif OPTICS_CASE == "brightfield":

            real_noise = dt.Gaussian(
                mu=0, 
                sigma=lambda: np.random.rand() * NOISE_BRIGHTFIELD_REAL,
            )
            noise = real_noise >> dt.Gaussian(
                mu=0, 
                sigma=lambda real_sigma: real_sigma * NOISE_BRIGHTFIELD_IMAG* 1j,
                real_sigma=real_noise.sigma
            )
            training_data = training_data >> noise
    
    if OPTICS_CASE == "brightfield":
        training_data = (training_data >> dt.Real()) & (training_data >> dt.Imag())
        training_data = training_data >> dt.Merge(lambda: lambda x: np.concatenate( [np.array(_x) for _x in x], axis=-1 ))

    #To get the labels
    training_data.store_properties()

    #Generate the data
    frame = training_data.update().resolve()

    #Get the labels
    labels = get_labels(training_data)

    #Transform the data into a numpy array
    frame = np.array(frame, dtype = np.float32)

    #Save the data
    np.save(f"../data/{OPTICS_CASE}_data.npy", frame)
    np.save(f"../data/{OPTICS_CASE}_labels.npy", labels)
    plt.imsave(f"../assets/{OPTICS_CASE}_frame.png", frame[...,-1])

if __name__ == "__main__":
    main()