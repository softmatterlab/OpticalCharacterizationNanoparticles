# Content in utilites folder

* `generate_data.py` - Script for simulating data and labels for the three different microscopy techniques.

* `helpers.py` - Script containing all helper functions for the notebooks.

* `rvt.py` - Script for Efficient Python implementation of Radial Variance Transform. 


## Generating a experimental frame and labels for the three different microscopy techniques (`generate_data.py`)

This script simulates imaging of particles using three different optical setups: holography, darkfield, and iSCAT. It defines optical and particle parameters, generates synthetic image data with noise, and saves the images and labels for further analysis using the DeepTrack library.


## Helper Functions for Task Analysis in Example Notebooks (`helpers.py`)

This python script contains a set of helper functions for analyzing different tasks in the example notebooks. Below is a list of the available functions and their descriptions.

### Functions

- **`rvt_pipeline`**: Applies the Radial Variance Transform (RVT) to an image and returns the detected particles.
- **`add_bin_circles`**: Adds circles to an image based on specified parameters.
- **`get_rois`**: Retrieves regions of interest (ROIs) from the input data.
- **`get_F1_score`**: Computes the F1 score between the predicted particles and the ground truth data.
- **`get_polarizability_rr`**: Calculates the polarizability of a particle based on given parameters.
- **`get_polarizability`**: Calculates the polarizability of a particle with a specified medium refractive index.
- **`form_factor`**: Calculates the spherical form factor in nanometers.
- **`signal_iscat`**: Calculates the signal in interferometric scattering microscopy (iSCAT).
- **`darkfield_intensity`**: Computes the darkfield intensity captured by a camera.
- **`darkfield_intensity_range`**: Calculates the range of darkfield intensity values captured by the camera.
- **`pol_range`**: Calculates the range of polarizabilities given arrays of possible radii and refractive indices.
- **`pol_range_mean_std`**: Computes the mean and standard deviation of polarizabilities given arrays of possible radii and refractive indices.
- **`signal_range`**: Computes the range of signals based on arrays of possible radii and refractive indices.
- **`signal_range_mean_std`**: Calculates the mean and standard deviation of signals given arrays of possible radii and refractive indices.
- **`plot_frame_with_detections`**: Plots a frame with the detected particles.
- **`plot_frame_with_detections_filled`**: Plots a frame with filled detections of particles.
- **`plot_overlay`**: Overlays the ground truth and detected particles on a single image for comparison.
- **`visualize_lab_pred`**: Visualizes the predictions made on the validation set.
- **`gaussian_fit`**: Fits a 2D Gaussian to the input data and returns the resulting Gaussian values.
- **`radial_variance_gaussian`**: Creates a score map using the Radial Variance Transform and fits a 2D Gaussian to the data.


## Efficient Python implementation of Radial Variance Transform (`rvt.py`)


The main function is :func:`rvt` in the bottom of the file, which applies the transform to a single image (2D numpy array).


Compared to the vanilla convolution implementation, there are two speed-ups:
1) Pre-calculating and caching kernel FFT; this way so only one inverse FFT is calculated per convolution + one direct fft of the image is used for all convolutions
2) When finding MoV, calculate ``np.mean(rsqmeans)`` in a single convolution by averaging all kernels first

### Functions

- **`gen_r_kernel`**: Generate a ring kernel with radius `r` and size ``2*rmax+1``
- **`generate_all_kernels`**: Generate a set of kernels with radii between `rmin` and `rmax` and sizes ``2*rmax+1``
- **`_check_core_args`**: Check validity of the core algorithm arguments
- **`_check_args`**: Check validity of all the algorithm arguments
- **`get_fshape`**: Get the required shape of the transformed image given the shape of the original image and the kernel
- **`prepare_fft`**: Prepare the image for a convolution by taking its Fourier transform, applying padding if necessary
- **`convolve_fft`**: Calculate the convolution from the Fourier transforms of the original image and the kernel, trimming the result if necessary
- **`rvt_core`**: Perform core part of Radial Variance Transform (RVT) of an image
- **`high_pass`**: Perform Gaussian high-pass filter on the image
- **`rvt`**: Perform Radial Variance Transform (RVT) of an image 