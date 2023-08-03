import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import numpy as np

# Load the NIfTI file containing label information
file_path = "/home/fangj1/Code/Vision-Language-on-3D-Scene-Understanding/src/data/bcv/RawData/Training/label/label0004.nii.gz"
label_nifti = nib.load(file_path)

# Access the label data as a NumPy array
label_data = label_nifti.get_fdata()


# Find the index of the slice with the maximum value
max_value_slice_idx = np.argmax(np.max(label_data, axis=(0, 1)))

# Print the index of the slice with the maximum value
print("Index of the slice with the maximum value:", max_value_slice_idx)
