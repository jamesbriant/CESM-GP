import argparse


def get_base_parser():
    parser = argparse.ArgumentParser(description="Train a Gaussian Process model.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing the NetCDF files.",
    )
    # parser.add_argument(
    #     "--model-type",
    #     type=str,
    #     choices=["independent", "shared", "multioutput"],
    #     default="independent",
    #     help="Type of GP model to use.",
    # )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=50,
        help="Number of training iterations.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
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
    return parser
