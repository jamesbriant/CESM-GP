import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(description="Train a Gaussian Process model.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the NetCDF files.",
    )
    parser.add_argument(
        "--target_var",
        type=str,
        choices=["temp", "qv"],
        required=True,
        help="Name of the variable to be used as the target.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        required=True,
        help="Number of samples to draw from the dataset. The format of this argument depends on the requirements of the chosen sampler.",
    )
    parser.add_argument(
        "--training_iterations",
        type=int,
        default=50,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for the optimizer.",
    )
    # parser.add_argument(
    #     "--batch-size",
    #     type=int,
    #     default=32,
    #     help="Batch size for training.",
    # )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save outputs and models.",
    )
    parser.add_argument(
        "--min_pfull",
        type=float,
        default=0.0,
        help="Minimum pfull value to filter the data.",
    )
    return parser
