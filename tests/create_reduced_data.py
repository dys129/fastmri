import h5py
import fastmri
from fastmri.data.transforms import complex_center_crop, to_tensor, tensor_to_complex_np
import numpy as np
from pathlib import Path

num_middle_slices=15
size=64

source_data_folder="/home/anon/reduced_fastmri/singlecoil_train"
dest_data_folder="/home/anon/reduced_fastmri/small_data/singlecoil_train"

# source_data_folder="/home/anon/reduced_fastmri/singlecoil_val"
# dest_data_folder="/home/anon/reduced_fastmri/small_data/singlecoil_val"

# source_data_folder="/home/anon/reduced_fastmri/singlecoil_test"
# dest_data_folder="/home/anon/reduced_fastmri/small_data/singlecoil_test"

if __name__ == '__main__':
    files = list(Path(source_data_folder).iterdir())
    dest_path = Path(dest_data_folder)
    for fname in sorted(files):
        print (fname)
        orig = h5py.File(fname)
        dest = h5py.File(dest_path / fname.name, 'a')
        volume_kspace = orig['kspace'][()]
        kspace_list = []
        reconstruction_list = []

        total_slices = volume_kspace.shape[0]
        for i in range(total_slices//2 - num_middle_slices//2, total_slices//2 + num_middle_slices//2 + 1):
            slice_kspace = volume_kspace[i]
            slice_kspace2 = to_tensor(slice_kspace)  # Convert from numpy array to pytorch tensor
            kspace_crop = complex_center_crop(slice_kspace2, (size, size))
            ift = fastmri.ifft2c(kspace_crop)  # Apply Inverse Fourier Transform to get the complex image
            reconstruction = fastmri.complex_abs(ift)
            kspace_list.append(tensor_to_complex_np(kspace_crop))
            reconstruction_list.append(reconstruction)

        dest['kspace'] = np.stack(kspace_list)
        dest['reconstruction_esc'] = np.stack(reconstruction_list)
        dest['ismrmrd_header']=orig['ismrmrd_header'][()]

        dest.close()
        orig.close()

