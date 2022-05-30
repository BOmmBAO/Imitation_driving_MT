#!/usr/bin/env python


# x y z pitch(y) row(x) yaw(z)
def train_coordinates(task_mode):
    starts = {
        'Straight': [[322.09, 129.50, 0.5, 0.0, 0.0, 180.0],
                     [88.13, 4.32, 0.5, 0.0, 0.0, 90.0],
                     [392.47, 87.41, 0.5, 0.0, 0.0, 90.0],
                     [383.18, -2.20, 0.5, 0.0, 0.0, 180.0],
                     [283.67, 129.48, 0.5, 0.0, 0.0, 180.0]],
        'U_curve': [
            [92.95, 55.55, 0.5, 0.0, 0.0, -147.17],
        ],
        'Curve': [[151.60, 83.00, 0.5, 0.0, 0.0, 90.0],
                  [84.16, 144.85, 0.5, 0.0, 0.0, 353.20],
                  [25, 326.54, 0.5, 0.0, 0.0, 180.0],
                  [-2., 302, 0.5, 0.0, 0.0, 90.0]],
        'Long': [
            [58.13, -2.04, 0.5, 0.0, 0.0, 180.0],
            [48.13, 326.57, 0.5, 0.0, 0.0, 180.0],
            [355.0, 330.61, 0.5, 0.0, 0.0, 0.0],
            [350.0, 2, 0.5, 0.0, 0.0, 0.0],
            [108.13, 105.43, 0.5, 0.0, 0.0, 180.0],
            [-3.46, 170.01, 0.5, 0.0, 0.0, 270.0],
            [80, 306.56, 0.5, 0.0, 0.0, 0.0],
            [189.69, 259.42, 0.5, 0.0, 0.0, 90.0],
        ],
        'Lane': [
            [58.07, -186.70, 0.5, 0.0, 0.0, 359.76],
            [60.13, -190.21, 0.5, 0.0, 0.0, 359.76],
            [108.07, 201.77, 0.5, 0.0, 0.0, -4.43],
            [108.03, 205.27, 0.5, 0.0, 0.0, -4.27],
        ],
        'Lane_test': [[229.18, 100.16, 0.5, 0.0, 0.0, 90.0],
                      [232.68, 100.16, 0.5, 0.0, 0.0, 90.0],
                      [58.13, 203.89, 0.5, 0.0, 0.0, 0.0],
                      [58.13, 207.39, 0.5, 0.0, 0.0, 0.0]],
    }

    dests = {
        'Straight': [[110.47, 129.50, 0.5, 0.0, 0.0, 180.0],
                     [88.13, 299.92, 0.5, 0.0, 0.0, 90.0],
                     [392.47, 308.21, 0.5, 0.0, 0.0, 90.0],
                     [185.55, -1.95, 0.5, 0.0, 0.0, 180.0],
                     [128.94, 129.75, 0.5, 0.0, 0.0, 180.0]],
        'U_curve': [[92.74, 64.81, 0.5, 0.0, 0.0, -30.44]],
        'Curve': [[86.30, 141.05, 0.5, 0.0, 0.0, 172.13],
                  [155.09, 85.60, 0.5, 0.0, 0.0, 270.0],
                  [2.01, 300, 0.5, 0.0, 0.0, 270.0],
                  [23, 330.54, 0.5, 0.0, 0.0, 0.0]],
        'Long': [
            [68.13, 330.57, 0.5, 0.0, 0.0, 0.0],
            [80.13, 1.96, 0.5, 0.0, 0.0, 0.0],
            [355.0, -2.00, 0.5, 0.0, 0.0, 180.0],
            [350.0, 326.57, 0.5, 0.0, 0.0, 180.0],
            [-7.43, 150.01, 0.5, 0.0, 0.0, 90.0],
            [88.13, 109.42, 0.5, 0.0, 0.0, 0.0],
            [193.69, 259.42, 0.5, 0.0, 0.0, 270.0],
            [80, 302.54, 0.5, 0.0, 0.0, 180.0],
        ],
        'Lane': [[68.09, 187.91, 0.5, 0.0, 0.0, 180.29],
                 [70.09, 191.42, 0.5, 0.0, 0.0, 180.29],
                 [78.06, -200.78, 0.5, 0.0, 0.0, 179.76],
                 [78.06, -204.28, 0.5, 0.0, 0.0, 179.76]],
        'Lane_test': [[58.13, 193.39, 0.5, 0.0, 0.0, 180.0],
                      [58.13, 196.89, 0.5, 0.0, 0.0, 180.0],
                      [239.68, 100.16, 0.5, 0.0, 0.0, 270.0],
                      [243.18, 100.16, 0.5, 0.0, 0.0, 270.0]],
    }
    return starts[task_mode], dests[task_mode]