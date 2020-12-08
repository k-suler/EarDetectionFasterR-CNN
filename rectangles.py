import pdb

import numpy as np

arr = [
    [
        0,
    ]
    * 10,
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [
        0,
    ]
    * 10,
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [
        0,
    ]
    * 10,
]


def find_rectangles(arr):
    # Deeply copy the array so that it can be modified safely
    arr = [row[:] for row in arr]

    rectangles = []

    for top, row in enumerate(arr):
        start = 0

        # Look for rectangles whose top row is here
        while True:
            try:
                left = row.index(1, start)
            except ValueError:
                break

            # Set start to one past the last 0 in the contiguous line of 0s
            try:
                start = row.index(0, left)
            except ValueError:
                start = len(row)

            right = start - 1

            if (  # Width == 1
                left == right
                or
                # There are 0s above
                top > 0
                and all(arr[top - 1][left : right + 1])
            ):
                continue

            bottom = top + 1
            while (
                bottom < len(arr)
                and
                # No extra ones on the sides
                (left == 0 or not arr[bottom][left - 1])
                and (right == len(row) - 1 or not arr[bottom][right + 1])
                and
                # All zeroes in the row
                all(arr[bottom][left : right + 1])
            ):
                print(bottom)
                bottom += 1
            # The loop ends when bottom has gone too far, so backtrack
            bottom -= 1

            if (  # Height == 1
                bottom == top
                or
                # There are 0s beneath
                (bottom < len(arr) - 1 and all(arr[bottom + 1][left : right + 1]))
            ):
                continue

            rectangles.append((top, left, bottom, right))

            # Remove the rectangle so that it doesn't affect future searches
            for i in range(top, bottom + 1):
                arr[i][left : right + 1] = [0] * (right + 1 - left)
    return rectangles


from pprint import pprint

pprint(arr)

print(find_rectangles(arr))

a2 = np.array(arr)

pdb.set_trace()


def find_rectangles(self, arr):
    # Deeply copy the array so that it can be modified safely
    arr = [row[:] for row in arr]

    rectangles = []

    for top, row in enumerate(arr):
        start = 0

        # Look for rectangles whose top row is here
        while True:
            try:
                left = row.index(1, start)
            except ValueError:
                break

            # Set start to one past the last 0 in the contiguous line of 0s
            try:
                start = row.index(0, left)
            except ValueError:
                start = len(row)

            right = start - 1

            if (  # Width == 1
                left == right
                or
                # There are 0s above
                top > 0
                and all(arr[top - 1][left : right + 1])
            ):
                continue

            bottom = top + 1
            while (
                bottom < len(arr)
                and
                # No extra ones on the sides
                (left == 0 or not arr[bottom][left - 1])
                and (right == len(row) - 1 or not arr[bottom][right + 1])
                and
                # All zeroes in the row
                all(arr[bottom][left : right + 1])
            ):
                bottom += 1
            # The loop ends when bottom has gone too far, so backtrack
            bottom -= 1

            if (  # Height == 1
                bottom == top
                or
                # There are 0s beneath
                (bottom < len(arr) - 1 and all(arr[bottom + 1][left : right + 1]))
            ):
                continue

            rectangles.append([left, top, right, bottom])

            # Remove the rectangle so that it doesn't affect future searches
            for i in range(top, bottom + 1):
                arr[i][left : right + 1] = [0] * (right + 1 - left)
    return rectangles