import h5py
import fastmri
from fastmri.data.transforms import complex_center_crop, to_tensor, tensor_to_complex_np
import numpy as np
from pathlib import Path

num_middle_slices=15
size=64

source_data_folder="/home/anon/reduced_fastmri/singlecoil_train"
dest_data_folder="/home/anon/reduced_fastmri/small_train"

if __name__ == '__main__':
    files = list(Path(source_data_folder).iterdir())
    dest_path = Path(dest_data_folder)
    for fname in sorted(files):
        print (fname)
        orig = h5py.File(fname)
        hf = h5py.File(dest_path / fname.name, 'a')
        volume_kspace = orig['kspace'][()]
        kspace_list = []
        reconstruction_list = []

        total_slices = volume_kspace.shape[0]
        for i in range(total_slices//2 - num_middle_slices//2, total_slices//2 + num_middle_slices//2 + 1):
            slice_kspace = volume_kspace[i]
            slice_kspace2 = to_tensor(slice_kspace)  # Convert from numpy array to pytorch tensor
            slice_image = fastmri.ifft2c(slice_kspace2)  # Apply Inverse Fourier Transform to get the complex image
            slice_crop = complex_center_crop(slice_kspace2, (size, size))
            image_crop = complex_center_crop(slice_image, (size, size))
            reconstruction = fastmri.complex_abs(image_crop)

            kspace_list.append(tensor_to_complex_np(slice_crop))
            reconstruction_list.append(reconstruction)

        hf['kspace'] = np.stack(kspace_list)
        hf['reconstruction_esc'] = np.stack(reconstruction_list)

        hf.close()
        orig.close()

        break