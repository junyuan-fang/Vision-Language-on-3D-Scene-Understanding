import nibabel as nib
import matplotlib.pyplot as plt

# Specify the file path to img0001.nii.gz
file_path = "/home/fangj1/Code/Vision-Language-on-3D-Scene-Understanding/src/data/bcv/RawData/Training/img/img0002.nii.gz"

# Load the NIfTI file
img_nifti = nib.load(file_path)

# Access the image data as a NumPy array
img_data = img_nifti.get_fdata()

# Print some information about the image
print("Image Shape:", img_data.shape)#Image Shape: (512, 512, 147),Image Shape: (512, 512, 139) or ...
print("Image Data Type:", img_data.dtype)

# # Select the slice number you want to plot (e.g., slice_idx = 0 for the first slice)
# slice_idx = 0

# # Get the 2D slice from the 3D volume
# img_slice = img_data[:, :, slice_idx]
# # Plot the slice
# plt.imshow(img_slice, cmap='gray')
# plt.title(f"Slice {slice_idx+1} of img0001.nii.gz")
# plt.axis('off')
# plt.show()


start_slice_idx = 0

# Plot 10 consecutive slices
num_slices_to_plot = 10

fig, axes = plt.subplots(2, 5, figsize=(15, 8))

for i in range(num_slices_to_plot):
    slice_idx = start_slice_idx + i
    img_slice = img_data[:, :, slice_idx]

    # Plot the slice
    row = i // 5
    col = i % 5
    axes[row, col].imshow(img_slice, cmap='gray')
    axes[row, col].set_title(f"Slice {slice_idx + 1}")
    axes[row, col].axis('off')

plt.suptitle("Consecutive Slices of img0001.nii.gz", fontsize=16)
plt.show()