from pathlib import Path
import argparse

import numpy as np

from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform

def main(args):
    root = args.root

    # print
    df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    print("Before clean and balance:")
    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

    # clean
    df = read_df(root)
    df.drop(df[df["x"] < 0.02].index, inplace=True)
    df.drop(df[df["y"] < 0.02].index, inplace=True)
    df.drop(df[df["z"] < 0.02].index, inplace=True)
    df.drop(df[df["x"] > 0.28].index, inplace=True)
    df.drop(df[df["y"] > 0.28].index, inplace=True)
    df.drop(df[df["z"] > 0.28].index, inplace=True)
    # write_df(df, root)

    # balance
    # df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    i = np.random.choice(negatives.index, len(negatives.index) - len(positives.index), replace=False)
    df = df.drop(i)
    write_df(df, root)

    # remove unreferenced scenes.
    # df = read_df(root)
    scenes = df["scene_id"].values
    for f in (root / "scenes").iterdir():
        if f.suffix == ".npz" and f.stem not in scenes:
            print("Removed", f)
            f.unlink()

    # print
    df = read_df(root)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    print("After clean and balance:")
    print("Number of samples:", len(df.index))
    print("Number of positives:", len(positives.index))
    print("Number of negatives:", len(negatives.index))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    main(args)