import unittest
import torch
import numpy as np

from samplers import RandomSampler, LatinHypercubeSampler, StratifiedSampler

class MockDataset:
    def __init__(self, num_times, num_lats, num_lons):
        self.num_times = num_times
        self.num_lats = num_lats
        self.num_lons = num_lons
        self.lat_lon_size = self.num_lats * self.num_lons
        self.length = self.num_times * self.lat_lon_size

    def __len__(self):
        return self.length

    def indices_to_idx(self, time_idx, lat_idx, lon_idx):
        return (time_idx * self.lat_lon_size) + (lat_idx * self.num_lons) + lon_idx

class TestSamplers(unittest.TestCase):

    def setUp(self):
        self.dataset = MockDataset(num_times=10, num_lats=20, num_lons=30)
        self.n_samples = 50

    def test_random_sampler(self):
        sampler = RandomSampler(self.dataset, self.n_samples)
        samples = list(sampler)
        self.assertEqual(len(samples), self.n_samples)
        self.assertEqual(len(set(samples)), self.n_samples) # Check for uniqueness
        for sample in samples:
            self.assertGreaterEqual(sample, 0)
            self.assertLess(sample, len(self.dataset))

    def test_lhd_sampler(self):
        sampler = LatinHypercubeSampler(self.dataset, self.n_samples)
        samples = list(sampler)
        self.assertEqual(len(samples), self.n_samples)
        for sample in samples:
            self.assertGreaterEqual(sample, 0)
            self.assertLess(sample, len(self.dataset))

    def test_stratified_sampler_lon(self):
        sampler = StratifiedSampler(self.dataset, self.n_samples, strata=['lon'])
        samples = list(sampler)
        # n_samples might not be perfectly divisible, so we check the length is close
        self.assertAlmostEqual(len(samples), self.n_samples, delta=self.dataset.num_lons)
        for sample in samples:
            self.assertGreaterEqual(sample, 0)
            self.assertLess(sample, len(self.dataset))

    def test_stratified_sampler_time_lat(self):
        n_samples = 100
        sampler = StratifiedSampler(self.dataset, n_samples, strata=['time', 'lat'])
        samples = list(sampler)
        self.assertAlmostEqual(len(samples), n_samples, delta=self.dataset.num_times * self.dataset.num_lats)
        for sample in samples:
            self.assertGreaterEqual(sample, 0)
            self.assertLess(sample, len(self.dataset))

if __name__ == '__main__':
    unittest.main()
