import os
import random
import math
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

def plot_T2_slices(root_dir, ids_list, num_examples=9, cols=3, alpha=0.4):
    ids_to_plot = random.sample(ids_list, min(len(ids_list), num_examples))
    rows = math.ceil(len(ids_to_plot) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()

    for i, sample_id in enumerate(ids_to_plot):
        folder_path = os.path.join(root_dir, f"UCSF-PDGM-0{sample_id}_nifti")
        t2_file = os.path.join(folder_path, f"UCSF-PDGM-0{sample_id}_T2.nii.gz")
        seg_file = os.path.join(folder_path, f"UCSF-PDGM-0{sample_id}_tumor_segmentation.nii.gz")

        if not (os.path.exists(t2_file) and os.path.exists(seg_file)):
            axes[i].axis("off")
            continue

        t2_slice = nib.load(t2_file).dataobj[:, :, nib.load(t2_file).shape[2] // 2]
        seg_slice = nib.load(seg_file).dataobj[:, :, nib.load(seg_file).shape[2] // 2]

        axes[i].imshow(t2_slice.T, cmap="gray", origin="lower")
        axes[i].imshow(seg_slice.T, cmap="Reds", alpha=alpha, origin="lower")
        axes[i].set_title(f"Sample {sample_id}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def plot_training_results(losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(losses, label='Reconstruction Loss')
    ax1.set_title("Training Loss")
    ax1.legend()
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title("Classification Accuracy")
    ax2.legend()
    plt.show()