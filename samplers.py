import torch
import numpy as np
from scipy.stats import qmc
from torch.utils.data import Sampler as TorchSampler
from torch.utils.data import Dataset

class Sampler(TorchSampler):
    """
    Base class for all samplers.
    """
    def __init__(self, data_source: Dataset, n_samples: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.n_samples = n_samples

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.n_samples

class RandomSampler(Sampler):
    """
    Samples `n_samples` random points from the dataset.
    """
    def __iter__(self):
        return iter(torch.randperm(len(self.data_source))[:self.n_samples].tolist())

class LatinHypercubeSampler(Sampler):
    """
    Samples `n_samples` points using Latin Hypercube Sampling.
    """
    def __init__(self, data_source: Dataset, n_samples: int):
        super().__init__(data_source, n_samples)
        self.dims = {
            "time": self.data_source.num_times,
            "lat": self.data_source.num_lats,
            "lon": self.data_source.num_lons,
        }
        self.sampler = qmc.LatinHypercube(d=len(self.dims))

    def __iter__(self):
        sample = self.sampler.random(n=self.n_samples)
        l_bounds = [0, 0, 0]
        u_bounds = [self.dims["time"], self.dims["lat"], self.dims["lon"]]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds).astype(int)

        indices = [
            self.data_source.indices_to_idx(time_idx, lat_idx, lon_idx)
            for time_idx, lat_idx, lon_idx in sample_scaled
        ]
        return iter(indices)

class StratifiedSampler(Sampler):
    """
    Performs stratified sampling on the given dimensions.
    """
    def __init__(self, data_source: Dataset, n_samples: int, strata: list[str]):
        super().__init__(data_source, n_samples)
        self.strata = strata
        self.dims = {
            "time": self.data_source.num_times,
            "lat": self.data_source.num_lats,
            "lon": self.data_source.num_lons,
        }

    def __iter__(self):
        strata_dims = [self.dims[s] for s in self.strata]
        n_strata = np.prod(strata_dims)
        samples_per_stratum = self.n_samples // n_strata

        indices = []
        for i in range(n_strata):
            # This maps a flat stratum index to the multi-dimensional stratum
            temp_i = i
            stratum_key = {}
            for s_dim_name, s_dim_val in zip(self.strata, strata_dims):
                stratum_key[s_dim_name] = temp_i % s_dim_val
                temp_i //= s_dim_val

            non_strata_dims = {k: v for k, v in self.dims.items() if k not in self.strata}

            for _ in range(samples_per_stratum):
                sample_key = stratum_key.copy()
                for dim, dim_range in non_strata_dims.items():
                    sample_key[dim] = np.random.randint(0, dim_range)

                indices.append(self.data_source.indices_to_idx(sample_key["time"], sample_key["lat"], sample_key["lon"]))

        n_remaining = self.n_samples - len(indices)
        if n_remaining > 0:
            remaining_indices = torch.randperm(len(self.data_source))[:n_remaining].tolist()
            indices.extend(remaining_indices)

        return iter(indices)
