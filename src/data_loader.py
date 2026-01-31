import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class ROIDataset(Dataset):
    def __init__(self, roi_slices, labels):
        self.data = [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in roi_slices]
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def extract_tumor_slices(root_dir, subject_ids, sample_to_status, status_mapping):
    roi_slices = []
    labels = []

    for sample_id in tqdm(subject_ids, desc="Processing NIfTI volumes"):
        if sample_id not in sample_to_status: continue
        
        status_str = sample_to_status[sample_id].strip().lower()
        if status_str not in status_mapping: continue

        label = status_mapping[status_str]
        folder_path = os.path.join(root_dir, f"UCSF-PDGM-0{sample_id}_nifti")
        t2_file = os.path.join(folder_path, f"UCSF-PDGM-0{sample_id}_T2.nii.gz")
        seg_file = os.path.join(folder_path, f"UCSF-PDGM-0{sample_id}_tumor_segmentation.nii.gz")

        if not (os.path.exists(t2_file) and os.path.exists(seg_file)):
            continue

        t2_img = nib.load(t2_file).get_fdata()
        seg_img = nib.load(seg_file).get_fdata()

        for z in range(t2_img.shape[2]):
            # Create the ROI mask
            mask = seg_img[:, :, z] > 0
            if not np.any(mask):
                continue
                
            roi = t2_img[:, :, z] * mask
            
            if roi[roi > 0].mean() > 0.01 and (np.count_nonzero(roi)/roi.size) > 0.001:
                # normalization
                s_min, s_max = roi.min(), roi.max()
                denom = (s_max - s_min) if (s_max - s_min) > 0 else 1e-8
                norm_roi = (roi - s_min) / denom
                
                roi_slices.append(norm_roi.astype(np.float32))
                labels.append(label)
                    
    return roi_slices, labels