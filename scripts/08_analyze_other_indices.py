import rasterio
import numpy as np

file_path = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

with rasterio.open(file_path) as src:

    ndwi = src.read(11)
    mndwi = src.read(12)
    ndci = src.read(13)
    fai = src.read(14)

    print("INDEX DISTRIBUTION ANALYSIS")
    print("============================")

    indices = {
        "NDWI": ndwi,
        "MNDWI": mndwi,
        "NDCI": ndci,
        "FAI": fai
    }

    for name, data in indices.items():

        valid = data[np.isfinite(data)]

        print()
        print(name)
        print("--------------------")
        print("Minimum:", np.min(valid))
        print("Maximum:", np.max(valid))
        print("Mean:", np.mean(valid))
        print("Median:", np.median(valid))
        print("25th percentile:", np.percentile(valid, 25))
        print("75th percentile:", np.percentile(valid, 75))
        print("90th percentile:", np.percentile(valid, 90))
        print("95th percentile:", np.percentile(valid, 95))
        print("99th percentile:", np.percentile(valid, 99))