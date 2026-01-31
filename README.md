# Low-Grade Glioma Classification via Autoencoders

This repository contains a machine learning pipeline designed to predict the 1p/19q co-deletion mutation in patients with Low-Grade Gliomas (LGG) using non-invasive MRI radiomics.

The 1p/19q co-deletion is a critical biomarker for gliomas. Patients with this mutation typically show better responses to treatment and higher survival rates. Traditionally, this is diagnosed via invasive biopsies. This project shifts away from traditional, manually-engineered radiomics. Instead, it employs an unsupervised deep learning approach to discover latent imaging biomarkers in LGG. By training a Convolutional Autoencoder on multi-modal MRI data (UCSF-PDGM dataset), the model learns to compress complex tumor textures and geometries into a low-dimensional latent space. These learned features are then used to predict the 1p/19q co-deletion status, a critical prognostic marker in neuro-oncology.

## Project Structure
- `src/data_loader.py`: NIfTI processing and ROI extraction.
- `src/models.py`: Convolutional Autoencoder architecture.
- `src/utils.py`: Visualization tools.
- `main.py`: Full execution pipeline.

## Setup
1. Clone the repo.
2. Place the UCSF-PDGM dataset in a `data/` folder.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the pipeline: `python main.py`.