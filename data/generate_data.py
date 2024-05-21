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
RADIUS_RANGE = (100e-9, 200e-9)
REFRACTIVE_INDEX = (1.37, 1.6)
N_PARTICLES = 10
Z_RANGE = (-7.5, 7.5)

# Define the parameters of the noise - Need to be tuned
NOISE = True
NOISE_DARKFIELD = 0#1e-5
NOISE_ISCAT = 3e-3
NOISE_BRIGHTFIELD_REAL = 3e-2
NOISE_BRIGHTFIELD_IMAG = 3e-2

# Set the seed for reproducibility
np.random.seed(1234)

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
HC = dt.HorizontalComa(coefficient=lambda c1: c1, c1 = 0.3)
VC = dt.VerticalComa(coefficient=lambda c2:c2, c2 = 0.3)

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
            pupil=HC >> VC >> CROP
        )

    elif OPTICS_CASE == "darkfield":
        optics = dt.Darkfield(
            NA=NA,
            magnification=MAGNIFICATION,
            wavelength=WAVELENGTH,
            resolution=RESOLUTION,
            output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
            #coherence_length=1e-8,
            offset_z = 10,
            illumination_angle=0,
        )

    elif OPTICS_CASE == "iscat":
        optics = dt.ISCAT(
            NA=NA,
            magnification=MAGNIFICATION,
            wavelength=WAVELENGTH,
            resolution=RESOLUTION,
            output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
            illumination_angle=np.pi,
            pupil=HC >> VC >> CROP
        )

    #Define the particles
    particles = dt.MieSphere(
        radius=lambda: np.random.uniform(*RADIUS_RANGE),
        refractive_index=lambda: np.random.uniform(*REFRACTIVE_INDEX),
        position=lambda: np.random.uniform(20, IMAGE_SIZE-20, 2),
        z=lambda: np.random.uniform(*Z_RANGE),
        L=100) ^ N_PARTICLES
    
    #Define small noise for the particles inside optics
    if OPTICS_CASE == "brightfield":
        noise = dt.Gaussian(
            mu=0, 
            sigma=lambda: 1e-3*np.random.rand() + 1e-3*np.random.rand()*1j,
        )
    elif OPTICS_CASE == "iscat":
        noise = dt.Gaussian(
            mu=0, 
            sigma=lambda: 1e-3*np.random.rand(),
            )
    else:
        noise = dt.Gaussian(
            mu=0, 
            sigma=lambda: 1e-3*np.random.rand()*0,
        )

    #Define the optics and particles.
    training_data = optics(particles>>noise) 

    #Gaussian and poisson noise
    if NOISE == True:
        if OPTICS_CASE == "darkfield":
            training_data = training_data >> dt.Poisson(snr=lambda: 7 + np.random.rand() * 10, background=0) >> dt.Gaussian(sigma=lambda: np.random.rand() * NOISE_DARKFIELD)
        elif OPTICS_CASE == "iscat":
            training_data = training_data >> dt.Poisson(snr=lambda: 7 + np.random.rand() * 10, background=1) >> dt.Gaussian(sigma=lambda: np.random.rand() * NOISE_ISCAT)
        elif OPTICS_CASE == "brightfield":

            noise = dt.Gaussian(
                mu=0, 
                sigma=lambda: np.random.rand() * NOISE_BRIGHTFIELD_REAL + np.random.rand()*NOISE_BRIGHTFIELD_IMAG* 1j,
            )

            training_data = training_data >> noise
    
    #To get the labels
    training_data.store_properties()

    #Generate the data
    frame = training_data.update().resolve()

    #Get the labels
    labels = get_labels(frame)

    if OPTICS_CASE == "brightfield":
        new_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2), dtype = np.float32)
        new_frame[..., 0] = np.squeeze(np.real(frame))
        new_frame[..., 1] = np.squeeze(np.imag(frame))
        frame = new_frame

    #Transform the data into a numpy array
    frame = np.array(frame, dtype = np.float32)

    #Save the data
    print("Saving data...")
    np.save(f"../data/{OPTICS_CASE}_data.npy", frame)
    np.save(f"../data/{OPTICS_CASE}_labels.npy", labels)
    plt.imsave(f"../assets/{OPTICS_CASE}_frame.png", frame[...,-1], cmap = "gray")

if __name__ == "__main__":
    main()