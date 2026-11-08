import argparse
import os

import matplotlib.pyplot as plt
import xarray as xr

cutoff_values = {
    "temp": 0.1,
    "qv": 1e-4,
}
variable_names = {
    "temp": "Temperature",
    "qv": "Specific Humidity",
}


def main(variable: str, longitude: int, data_path: str):
    """
    Plots the standard deviation of a specified variable against latitude at a given longitude.

    Args:
        variable (str): The variable to plot (e.g., 'temp', 'qv').
        longitude (int): The longitude at which to extract the data.
        data_path (str): Path to the NetCDF data file.
    """
    # Load the dataset
    ds = xr.open_dataset(data_path)

    if variable not in ds:
        raise ValueError(f"Variable '{variable}' not found in the dataset.")

    # Select data at the specified longitude
    print("Extracting data for the first time slice ONLY.")
    ds = ds.isel(time=0)  # Assuming we want the first time slice
    data_at_lon = ds[variable].sel(lon=longitude, method="nearest")
    longitude_val = data_at_lon.lon.values

    # # Extract latitude and standard deviation values

    # Plotting
    print("Creating plots...")
    fig, axes = plt.subplots(
        # nrows=8, ncols=4, figsize=(10, 20), sharex=True, sharey="row"
        nrows=8,
        ncols=4,
        figsize=(8.3, 11.7),
        sharex=True,
        sharey="row",
    )
    axes = axes.flatten()
    for i, p in enumerate(ds.pfull.values):
        print(f"Plotting ({i}) for pressure level: {p} hPa")
        data_to_plot = data_at_lon.sel(pfull=p, method="nearest")
        data_to_plot.plot(ax=axes[i])
        axes[i].set_title(f"{p:.2f} hPa")
        axes[i].axhline(y=cutoff_values[variable], color="red", linestyle="--")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    plt.suptitle(
        f"Standard Deviation of {variable_names[variable]} at {longitude_val}° Longitude"
    )
    fig.supxlabel("Latitude (Degrees North)")
    fig.supylabel("Standard Deviation")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    file_name = f"{variable}_std_vs_lat_lon={longitude_val}_time=0.png"
    plt.savefig(f"figures/{file_name}")
    print(f"Plot saved as figures/{file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot standard deviation vs latitude for a given variable and longitude."
    )
    parser.add_argument(
        "--variable",
        type=str,
        required=True,
        help="The variable to plot (e.g., 'temp', 'qv')",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        required=True,
        help="The longitude at which to extract the data (e.g., -100.0)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the NetCDF data file",
    )

    args = parser.parse_args()
    main(args.variable, args.longitude, args.data_path)
