import rasterio
import numpy as np

file_path = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

with rasterio.open(file_path) as src:

    print("Total bands:", src.count)
    print("Band names:", src.descriptions)

    for band_number in range(11, 15):

        name = src.descriptions[band_number - 1]

        data = src.read(band_number)

        valid = data[np.isfinite(data)]

        print("\n----------------------------")
        print("Band:", band_number)
        print("Name:", name)
        print("Min:", valid.min())
        print("Max:", valid.max())
        print("Mean:", valid.mean())
        print("Valid pixels:", len(valid))