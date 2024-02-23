import h5py
file = "voxel_all.h5"
correct_predictions = 0
total_samples = 150
batch_size = 10

DATA_PATH = 'data/ScanObjectNN/object_dataset/'
READ_PATH = 'data/ScanObjectNN/processed_object_dataset/'



with h5py.File(READ_PATH+file, 'r') as f:
    visititems = f.visititems(print)


