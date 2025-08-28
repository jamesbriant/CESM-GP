import os
from typing import List

import torch
import xarray as xr
from torch.utils.data import Dataset


class NetCDFDataset(Dataset):
    """
    PyTorch Dataset for loading data from multiple NetCDF files.
    Uses xarray to lazily load data, making it suitable for large datasets.
    """

    def __init__(
        self,
        data_path: str,
        feature_vars: List[str],
        target_var,
        min_pfull: float = 0,
    ):
        """
        Args:
            data_path (str): Path to the directory containing the NetCDF files.
            feature_vars (list of str): Names of the variables to be used as features.
            target_var (str): Name of the variable to be used as the target.
            min_pfull (float): Minimum pfull value to filter the data.
        """
        assert target_var in ["temp", "qv"], "target_var must be either 'temp' or 'qv'."
        assert min_pfull >= 0, "min_pfull must be non-negative."

        super().__init__()
        # Use open_mfdataset to open multiple files as a single dataset.
        # chunks={} ensures that the data is loaded lazily.
        self.X = xr.open_mfdataset(
            [
                os.path.join(data_path, "temp_C3072_12288x6144.fre_mn_cesm.nc"),
                os.path.join(data_path, "qv_C3072_12288x6144.fre_mn_cesm.nc"),
            ],
            chunks={},
        )
        self.y = xr.open_dataset(
            os.path.join(data_path, f"{target_var}_C3072_12288x6144.fre_std_cesm.nc"),
            chunks={},
        )
        self.y = self.y.rename({target_var: f"{target_var}_std"})
        self.ds = xr.merge([self.X, self.y])

        if min_pfull > 0:
            # Filter the dataset based on the min_pfull value
            self.ds = self.ds.sel(pfull=slice(min_pfull, None))

        self.feature_vars = feature_vars
        self.target_var = target_var

        # Store the sizes of the dimensions we will iterate over
        self.num_times = len(self.ds.time)
        self.num_lats = len(self.ds.lat)
        self.num_lons = len(self.ds.lon)
        self.num_pfull = len(self.ds.pfull)

        # Pre-calculate the size of a 2D lat-lon slice and 3D time-lat-lon slice for efficiency
        self.lat_lon_size = self.num_lats * self.num_lons
        self.length = self.num_times * self.lat_lon_size

    def idx_to_indices(self, idx: int) -> tuple[int, int, int]:
        """
        Converts a single flat index into 3D (time, lat, lon) indices.

        Args:
            idx (int): The flat index in the range [0, len(dataset)-1].

        Returns:
            tuple[int, int, int]: A tuple containing (time_idx, lat_idx, lon_idx).
        """
        if not 0 <= idx < len(self):
            raise IndexError("Index out of range")

        # Find the time index by seeing how many full lat/lon slices fit into idx
        time_idx = idx // self.lat_lon_size

        # Find the remaining index within the current time slice
        idx_in_slice = idx % self.lat_lon_size

        # From the remainder, find the lat and lon indices
        lat_idx = idx_in_slice // self.num_lons
        lon_idx = idx_in_slice % self.num_lons

        return time_idx, lat_idx, lon_idx

    def indices_to_idx(self, time_idx: int, lat_idx: int, lon_idx: int) -> int:
        """
        Converts 3D (time, lat, lon) indices into a single flat index.

        Args:
            time_idx (int): The index for the 'time' dimension.
            lat_idx (int): The index for the 'latitude' dimension.
            lon_idx (int): The index for the 'longitude' dimension.

        Returns:
            int: The corresponding flat index.
        """
        if not (
            0 <= time_idx < self.num_times
            and 0 <= lat_idx < self.num_lats
            and 0 <= lon_idx < self.num_lons
        ):
            raise IndexError("One or more indices are out of range for the dimensions")

        # The formula for converting multi-dimensional indices to a flat index
        idx = (time_idx * self.lat_lon_size) + (lat_idx * self.num_lons) + lon_idx
        return idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Retrieves a single data point from the dataset.
        """
        # --- Convert 1D index to 3D (time, lat, lon) indices ---
        time_idx, lat_idx, lon_idx = self.idx_to_indices(idx)

        # --- Construct the GP input (train_x) and output (train_y) ---

        # Select the data for the specific (time, lat, lon) point.
        # This slice will contain all `pfull` values.
        sample_point = self.ds.isel(time=time_idx, lat=lat_idx, lon=lon_idx)

        # `train_x` is the humidity data along the `pfull` dimension
        inputs = sample_point[self.feature_vars].to_array(dim="features")
        train_x = torch.from_numpy(inputs.values.T).float()

        # `train_y` is the temperature data along the `pfull` dimension
        targets = sample_point[f"{self.target_var}_std"]
        train_y = torch.from_numpy(targets.values).float()

        # train_x will have shape [num_features, N_pfull]
        # train_y will have shape [N_pfull]
        return train_x, train_y
