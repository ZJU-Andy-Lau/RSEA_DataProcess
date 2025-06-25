import rasterio
import numpy as np
import argparse
import os

def convert_npy_to_geotiff(input_tif_path, input_npy_path, output_tif_path):
    """
    Converts a NumPy array (H, W) to a GeoTIFF file, preserving
    the georeferencing information from an existing GeoTIFF.

    Args:
        input_tif_path (str): Path to the reference GeoTIFF file
                              (to get CRS and transform).
        input_npy_path (str): Path to the input NumPy array file (.npy).
        output_tif_path (str): Path where the output GeoTIFF file will be saved.
    """
    try:
        # 1. Read the geometric information (CRS, transform) from the input TIFF
        with rasterio.open(input_tif_path) as src:
            profile = src.profile
            height = src.height
            width = src.width

        # 2. Load the NumPy array and replace NaNs with 0
        npy_data = np.load(input_npy_path)

        # Ensure the NumPy array dimensions match the reference TIFF
        if npy_data.shape != (height, width):
            raise ValueError(
                f"NumPy array shape {npy_data.shape} does not match "
                f"reference TIFF dimensions ({height}, {width})."
            )

        # Replace NaN values with 0
        npy_data = np.nan_to_num(npy_data, nan=0.0)

        # Update the profile for the output TIFF
        profile.update(
            dtype=npy_data.dtype,
            count=1,  # Assuming a single-band output TIFF
            nodata=0, # Optionally set nodata value to 0 if 0 represents no data
                      # (be careful if 0 is a valid data value)
        )

        # 3. Write the NumPy array to a new TIFF file with the same CRS and transform
        with rasterio.open(output_tif_path, 'w', **profile) as dst:
            dst.write(npy_data, 1) # Write the array as the first band

        print(f"Successfully converted '{input_npy_path}' to '{output_tif_path}' "
              f"using georeferencing from '{input_tif_path}'.")

    except FileNotFoundError as e:
        print(f"Error: One of the specified files was not found. {e}")
    except ValueError as e:
        print(f"Data error: {e}")
    except rasterio.errors.RasterioIOError as e:
        print(f"Rasterio I/O error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    options = parser.parse_args()
    file_list = os.listdir(options.root)
    npy1_path = [os.path.join(options.root,i) for i in file_list if 'res_1' in i][0]
    npy2_path = [os.path.join(options.root,i) for i in file_list if 'res_2' in i][0]
    tif1_path = [os.path.join(options.root,i) for i in file_list if 'overlap_1' in i][0]
    tif2_path = [os.path.join(options.root,i) for i in file_list if 'overlap_2' in i][0]
    convert_npy_to_geotiff(tif1_path,npy1_path,os.path.join(options.root,'res_vis_1.tif'))
    convert_npy_to_geotiff(tif2_path,npy2_path,os.path.join(options.root,'res_vis_2.tif'))