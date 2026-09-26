import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point

folder = Path("data/labels/hab_masks/training_waterbodies")
output = Path("results/reference_point_check.png")

waterbodies = {
    "Saroor Nagar": {
        "file": "Saroor_Nagar_boundary.geojson",
        "lat": 17.3539,
        "lon": 78.5270,
    },
    "Himayat Sagar": {
        "file": "Himayat_Sagar_boundary.geojson",
        "lat": 17.28233,
        "lon": 78.36119,
    },
    "Shamirpet Lake": {
        "file": "Shamirpet_Lake_boundary.geojson",
        "lat": 17.60967,
        "lon": 78.55663,
    },
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (name, info) in zip(axes, waterbodies.items()):

    gdf = gpd.read_file(folder / info["file"])

    # Plot waterbody
    gdf.plot(ax=ax)

    # Reference point
    point = gpd.GeoSeries(
        [Point(info["lon"], info["lat"])],
        crs="EPSG:4326"
    )

    point.plot(
        ax=ax,
        marker="x",
        markersize=120,
        linewidth=3
    )

    ax.set_title(name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Print whether point is inside
    inside = gdf.geometry.contains(
        Point(info["lon"], info["lat"])
    ).any()

    print(f"{name}:")
    print(f"  Reference point: {info['lat']}, {info['lon']}")
    print(f"  Inside polygon: {inside}")
    print()

plt.tight_layout()
plt.savefig(output, dpi=200)
plt.close()

print("=" * 60)
print(f"Saved: {output}")
print("=" * 60)