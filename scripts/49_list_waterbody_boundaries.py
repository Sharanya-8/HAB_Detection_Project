import os


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "labels",
    "hab_masks"
)


def main():

    print("=" * 70)
    print("STEP 49 - FIND ALL WATERBODY BOUNDARY FILES")
    print("=" * 70)

    print()
    print(f"Searching inside:")
    print(BASE_DIR)
    print()

    found = []

    for root, dirs, files in os.walk(BASE_DIR):

        for file in files:

            if file.lower().endswith(
                (".geojson", ".json", ".shp")
            ):

                full_path = os.path.join(
                    root,
                    file
                )

                found.append(full_path)

    if not found:

        print("No boundary/vector files found.")

    else:

        print(
            f"Found {len(found)} vector files:"
        )

        print()

        for path in sorted(found):

            relative = os.path.relpath(
                path,
                PROJECT_ROOT
            )

            print(relative)

    print()
    print("=" * 70)
    print("STEP 49 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()