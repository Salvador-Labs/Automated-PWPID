# 📦 Automatic-PWPID

Automatic Post-Watershed Phase-ID Segmentation (PWPID): automatically segments 3D, 3-phase microstructure images using watershed transform. 

---

## 🧠 Overview

This project performs 3D segmentation on 3-phase microstructure datasets where phases can be differentiated via their greyscale  values. The code processes 3D microstructure reconstructions and perform phase segmentation, where each voxel is labeled with 1,2, or 3 depending on its phase. It was developed and tested for segmenting 3-phase solid oxide cell (SOC) microstructures. This code requires files in the [.npy](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) file format. 

---

## 📁 Repository Structure

Automatic-PWPID/ 
├── main.py # Main segmentation script 
├── segmentation_utils.py # Helper functions 
├── NLM_FILTER.py # Script for running non-local means filtering
├── test_files/ # Example data for testing script 
├── requirements.txt # Dependencies 
└── README.md # Project documentation


