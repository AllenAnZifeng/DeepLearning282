import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--log_name",
        type=str,
        default=None,
        required=False,
        help="Name for this run"
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help="Path to params yaml file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=182,
        required=False,
        help="Path to params yaml file"
    )
    parser.add_argument(
        "--cuda",
        type=bool,
        default=True,
        help="Use cuda"
    )
    
    args = parser.parse_args()
    
    return args
    