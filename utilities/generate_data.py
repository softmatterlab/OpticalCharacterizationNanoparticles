import deeptrack as dt
import numpy as np
import os
import matplotlib.pyplot as plt

# Define the parameters of the optics
IMAGE_SIZE = 512
NA = 1
MAGNIFICATION = 1
WAVELENGTH = 532e-9
RESOLUTION = 1.14e-7
OPTICS_CASE = "holography"  # "holography", "darkfield", "iscat"

# Define the parameters of the particles
RADIUS_RANGE = (100e-9, 200e-9)
REFRACTIVE_INDEX = (1.37, 1.6)
N_PARTICLES = 100
Z_RANGE = (-7.5, 7.5)

# Define the parameters of the noise - Need to be tuned
NOISE = True
NOISE_DARKFIELD = 1e-4
NOISE_ISCAT = 1e-3
NOISE_QF_REAL = 3e-2
NOISE_QF_IMAG = 3e-2

# Save or not the data
SAVE = False

# Set the seed for reproducibility
np.random.seed(1234)


# Define the pupil function for noise in holography and ISCAT
def crop(pupil_radius):
    """ Crop the pupil function to a circle of radius pupil_radius.

    Args:
        pupil_radius (float): Radius of the pupil function.
    Returns:
        function: Function that crops the pupil function to
        a circle of radius pupil_radius.
    """

    def inner(image):
        x = np.arange(image.shape[0]) - image.shape[0] / 2
        y = np.arange(image.shape[1]) - image.shape[1] / 2
        X, Y = np.meshgrid(x, y)
        image[X ** 2 + Y ** 2 > pupil_radius ** 2] = 0
        return image
    return inner


CROP = dt.Lambda(crop, pupil_radius=lambda: 0.8*IMAGE_SIZE)

# Define the coma abberation
HC = dt.HorizontalComa(coefficient=lambda c1: c1, c1=0.3)
VC = dt.VerticalComa(coefficient=lambda c2: c2, c2=0.3)


def get_labels(image):
    """Get the labels of the particles in the image.

    Args:
        image (deeptrack.Image): Image containing the particles.

    Returns:
        np.ndarray: Array containing the labels of the particles.
    """

    array = np.zeros((N_PARTICLES, 5), dtype=np.float32)
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

    print(f"Generating {OPTICS_CASE} data...")
    # Define the optics
    if OPTICS_CASE == "holography":
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

    # Define the particles
    particles = dt.MieSphere(
        radius=lambda: np.random.uniform(*RADIUS_RANGE),
        refractive_index=lambda: np.random.uniform(*REFRACTIVE_INDEX),
        position=lambda: np.random.uniform(20, IMAGE_SIZE-20, 2),
        z=lambda: np.random.uniform(*Z_RANGE),
        L=100) ^ N_PARTICLES
    # Define small noise for the particles inside optics
    if OPTICS_CASE == "holography":
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
            sigma=lambda: 1e-3*np.random.rand(),
        )

    # Define the optics and particles.
    training_data = optics(particles >> noise)

    # Gaussian and poisson noise
    if NOISE:
        if OPTICS_CASE == "darkfield":
            training_data = training_data >> dt.Gaussian(
                sigma=lambda: np.random.rand() * NOISE_DARKFIELD
                )
        elif OPTICS_CASE == "iscat":
            training_data = training_data >> dt.Gaussian(
                sigma=lambda: np.random.rand() * NOISE_ISCAT
                )
        elif OPTICS_CASE == "holography":
            noise = dt.Gaussian(
                mu=0,
                sigma=lambda: np.random.rand() * NOISE_QF_REAL +
                np.random.rand() * NOISE_QF_IMAG * 1j,
            )
            training_data = training_data >> noise

    # To get the labels
    training_data.store_properties()

    # Generate the data
    frame = training_data.update().resolve()

    # Get the labels
    labels = get_labels(frame)

    if OPTICS_CASE == "holography":
        new_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 2), dtype=np.float32)
        new_frame[..., 0] = np.squeeze(np.real(frame))
        new_frame[..., 1] = np.squeeze(np.imag(frame))
        frame = new_frame

    # Transform the data into a numpy array
    frame = np.array(frame, dtype=np.float32)

    if SAVE:
        # Save the data
        print("Saving data...")

        # Paths to save the data
        data_path = os.path.join(
            "..", f"{OPTICS_CASE}", "data", f"{OPTICS_CASE}_data.npy"
            )
        labels_path = os.path.join(
            "..", f"{OPTICS_CASE}", "data", f"{OPTICS_CASE}_labels.npy"
            )
        image_path = os.path.join(
            "..", f"{OPTICS_CASE}", "data", f"{OPTICS_CASE}_frame.png"
            )

        # Save the data
        np.save(data_path, frame)
        np.save(labels_path, labels)
        plt.imsave(image_path, frame[..., -1], cmap="gray")


if __name__ == "__main__":
    main()
