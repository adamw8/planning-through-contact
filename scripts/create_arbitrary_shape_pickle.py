import pickle

import numpy as np


def main():
    output_file = "arbitrary_shape_pickles/snake.pkl"
    boxes = [
        {
            "name": "box",
            "size": [0.04064, 0.165, 0.040875],
            "transform": np.eye(4),
        },
        {
            "name": "box",
            "size": [0.165 / 2, 0.04064, 0.040875],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, 0.165 / 4],
                    [0.0, 1.0, 0.0, 0.165 / 2 - 0.04064 / 2],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
        {
            "name": "box",
            "size": [0.165 / 2, 0.04064, 0.040875],
            "transform": np.array(
                [
                    [1.0, 0.0, 0.0, -0.165 / 4],
                    [0.0, 1.0, 0.0, -0.165 / 2 + 0.04064 / 2],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        },
    ]

    with open(output_file, "wb") as f:
        pickle.dump(boxes, f)


if __name__ == "__main__":
    main()
