# ============================================================
# STEP 1: GOOGLE EARTH ENGINE AUTHENTICATION
# ============================================================

import ee


PROJECT_ID = "remote-sensing-project"


def authenticate_gee():
    """
    Try to initialize Google Earth Engine.

    The project can continue using previously downloaded
    Sentinel-2 data if GEE initialization is unavailable.
    """

    print("=" * 60)
    print("GOOGLE EARTH ENGINE AUTHENTICATION")
    print("=" * 60)

    try:
        ee.Initialize(project=PROJECT_ID)

        print("Earth Engine initialized successfully.")
        print(f"Project: {PROJECT_ID}")
        print("Authentication status: AVAILABLE")

        return True

    except Exception as error:

        print("Earth Engine is currently unavailable.")
        print()
        print("Reason:")
        print(error)
        print()
        print(
            "The project can continue using the existing "
            "Sentinel-2 data downloaded from Google Earth Engine."
        )
        print()
        print("Authentication status: UNAVAILABLE")

        return False


if __name__ == "__main__":

    gee_available = authenticate_gee()

    if gee_available:
        print()
        print("GEE authentication completed.")
    else:
        print()
        print("Continuing with local Sentinel-2 data.")