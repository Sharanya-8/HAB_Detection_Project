import matplotlib
matplotlib.use("Agg")

import rasterio
import matplotlib.pyplot as plt

file_path = "data/raw/sentinel2/HAB_Sentinel2_2023_2025.tif"

with rasterio.open(file_path) as src:
    indices = {
        "NDWI": 11,
        "MNDWI": 12,
        "NDCI": 13,
        "FAI": 14
    }

    for name, band_number in indices.items():
        data = src.read(band_number)

        plt.figure(figsize=(10, 8))
        plt.imshow(data)
        plt.colorbar(label=name)
        plt.title(f"{name} - HAB Detection")
        plt.axis("off")

        output_path = f"data/processed/visualizations/{name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {output_path}")