from torch.utils.data import DataLoader

from dataset import NetCDFDataset

# from samplers import LatinHypercubeSampler
from samplers import RandomSampler


def main(
    data_path: str,
    min_pfull: float = 0,
):
    """Train a batch independent multitask Gaussian Process model on synthetic data.
    Args:
        data_path (str): Path to the directory containing the NetCDF files.
        min_pfull (float): Minimum pfull value to filter the data.
    """
    n_samples = 100  # example value

    ds = NetCDFDataset(
        data_path=data_path,
        feature_vars=["temp", "qv"],
        target_var="temp",
        min_pfull=min_pfull,
        sample_size=n_samples,
    )

    num_pfull = ds.num_pfull
    print(f"Fitting the bottom {num_pfull} atmospheric levels.")

    # --- Build the sampler ---
    # sampler = LatinHypercubeSampler(ds, n_samples)
    sampler = RandomSampler(ds, n_samples)

    # --- Load a batch of data ---
    dl = DataLoader(ds, batch_size=n_samples, shuffle=False, sampler=sampler)
    train_x, train_y = next(iter(dl))  # ONLY CALL THIS ONCE TO GET A SINGLE BATCH

    print("train_x shape:", train_x.shape)
    print("train_y shape:", train_y.shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Gaussian Process model.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the NetCDF files.",
    )
    args = parser.parse_args()
    main(
        data_path=args.data_path,
        min_pfull=700.0,
    )
