import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to text file for np.loadtxt")
    args = parser.parse_args()
    data = np.loadtxt(args.input)
    print(data.shape)


if __name__ == "__main__":
    main()
