# Detection and Quantification in ISCAT

In this notebook we will investigate how to detect and quantify particles in ISCAT. 

The notebook contains the following sections:

1. Imports 
    - Importing the packages needed to run the code.
2. Detection in ISCAT
    - We leverage two different methods for particle detection: the Radial Variance Transform(RVT) and LodeSTAR.
3. Quantification of particle properties in ISCAT
    - Here we show how to simulate particles in the ISCAT regime and train a Convolutional Neural Network(CNN) for the quantification task.
    - We provide figures of how the estimated signal scales with fixed radius aswell as with fixed refractive index and corresponding theoretical curves.
4. Combining detection and quantification in ISCAT
    - Here we provide a short example on how to analyze an entire frame by performing tracking with LodeSTAR and signal quantification with a CNN.


## Simulated data in the ISCAT regime:

<p float="left">
  <img src="assets/iscat_frame.png" alt="ISCAT frame" width="400" />
  <img src="assets/iscat_rois.png" alt="ISCAT ROIs" width="400"/>
</p>


## How to apply to your own data:

Down below is a description of how to modify the notebook for use of your own data.

1. Collect data and transform/save it into `numpy` arrays, that is, a file ending with `.npy`. Additional tip is to crop the frame so that it is a factor of 2 (application to the LodeSTAR detection model requires it to be downsampled 2 times)

2. For training a LodeSTAR detection model it is needed to use a minimum of one crop around a particle. It is adviced to choose a particle that is clearly visable and seperable from other particles, and that is descriptive of the ones that the model aims to detect. A good size for the ROI(Region Of Interest) is 32x32, 48x48 or 64x64. If the model fails, or the detection result is not sufficient enough: try to choose more particles in the training and/or change the thresholding/augmentations slightly. The settings in the notebook are a suitable starting point, but may need some finetuning to your own data.

3. For training a Convolutional Neural Network(CNN) for predicting the signal of particles it is adviced to start with the neural network architecture provided in the notebook. The only modifications needed is to change the simulation of the training data so that it fits your experimental setup. That is, to change parameters such as `wavelength`, `resolution`, `Numerical Aperature` of the optical system, and which range of `radius` and `refractive index` fits your data.