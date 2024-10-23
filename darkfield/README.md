# Detection and Quantification in Darkfield

In this notebook we will investigate how to detect and quantify particles in Darkfield. 

The notebook contains the following sections:

1. Imports 
    - Importing the packages needed to run the code.
2. Detection in Darkfield
    - We leverage two different methods for particle detection: the Radial Variance Transform(RVT) and LodeSTAR.
3. Quantification of particle properties in Darkfield
    - Here we show how to simulate particles in the Darkfield regime and train a Convolutional Neural Network(CNN) for the quantification task.
    - We provide figures of how the estimated signal scales with fixed radius aswell as with fixed refractive index and corresponding theoretical curves.
    - We also give an example on how changing the illumination angle effects these curves.
4. Combining detection and quantification in Darkfield
    - Here we provide a short example on how to analyze an entire frame by performing tracking with LodeSTAR and signal quantification with a CNN.

