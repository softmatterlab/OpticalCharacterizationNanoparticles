# Optical Label-Free Microscopy Characterization of Dielectric Nanoparticles

By Berenice García Rodríguez, Erik Olsén, Fredrik Skärberg, Giovanni Volpe, Fredrik Höök and Daniel Midtvedt.
You can find the full paper on arXiv: [Optical Label-Free Microscopy Characterization of Dielectric Nanoparticles](https://arxiv.org/abs/2409.11810)


## Description

In this tutorial we provide three notebooks for particle characterization in the following regimes: Holographic microscopy, Darkfield Microscopy, and ISCAT(Interferometric Scattering Microscopy). See the following notebooks:

* [ISCAT_analysis](iscat/ISCAT_analysis.ipynb) : Analyzing particles imaged in ISCAT.
* [Holography_analysis](holography/Holography_analysis.ipynb) : Analyzing particles imaged in Holographic microscopy. 
* [Darkfield_analysis](darkfield/Darkfield_analysis.ipynb) : Analyzing particles imaged in Darkfield Microscopy.

Additionally, there is a folder called `utilities` containing the python files `rvt.py`, `helpers.py` and `generate_data.py`. For more information about these files see: [README](utilities/README.md)


## Dependencies

To run the notebooks please install (https://github.com/DeepTrackAI/deeplay) and (https://github.com/DeepTrackAI/deeptrack) and its following dependencies.

### Installation

You can install Deeplay using pip:
```bash
pip install deeplay
```
or
```bash
python -m pip install deeplay
```

This will automatically install the required dependencies, including PyTorch and PyTorch Lightning. If a specific version of PyTorch is desired, it can be installed separately.

and Deeptrack using pip

```bash
pip install deeptrack
```
or
```bash
python -m pip install deeptrack
```


## Usage

The notebooks provided are fully ready to use and will run without any modification. Each modality comes with an "experimental" frame with corresponding labels (x, y, z, radius, refractive index), which is meant to give users a basic idea of how to work with their own data and perform analysis. 

**Using Your Own Data**

To use this code with your own data:

1. Load your custom experimental frame into the project.
2. Retrain the models using parameters that suit your specific experimental setup.

This flexible approach allows for easy adaptation of the existing code to your unique dataset and requirements. More details on how to use your own data can be found in the README-files for each corresponding modality.

README-files:
* [ISCAT](iscat/README.md)
* [Holography](holography/README.md)
* [Darkfield](darkfield/README.md)


## Citation
If you use this code for your research, please cite our paper: [Optical Label-Free Microscopy Characterization of Dielectric Nanoparticles](https://arxiv.org/abs/2409.11810)


## Funding
This work was partly supported by the H2020 European Research Council (ERC) Starting Grant ComplexSwimmers (Grant No. 677511), the Horizon Europe ERC Consolidator Grant MAPEI (Grant No. 101001267), the Knut and Alice Wallenberg Foundation (Grant No. 2019.0079), and the Swedish Research Council (VR, Grant No. 2019-05238).
