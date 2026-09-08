import rasterio
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


input_file = "data/labels/HAB_clean_labels.tif"

output_file = "data/processed/visualizations/HAB_clean_labels.png"


with rasterio.open(input_file) as src:

    label = src.read(1)


# Mask NoData
display = np.ma.masked_where(label == 255, label)


plt.figure(figsize=(10, 8))

plt.imshow(display)

plt.title("Cleaned HAB Pseudo-Labels")

plt.xlabel("Pixel")
plt.ylabel("Pixel")

plt.colorbar(
    label="Label (0 = Non-HAB, 1 = HAB)"
)

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("Clean label visualization created successfully.")

print()
print("Saved to:")
print(output_file)