import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import time

# ["location_x", "location_y", "rotation", "major_axis_length", "minor_axis_length", "MAE"]

def make_viz(array, deg, trained_model):

    for loc_x, loc_y in [(0,0), (0,10), (10,0), (10,10)]:
        x_rot = []
        y_min_ax = []
        z_err = []
        to_predict = []
        for row in array:
            if row[1] == loc_x and row[2] == loc_y:
                to_predict.append(row[:-1])
                x_rot.append(row[3])
                y_min_ax.append(row[5])
                z_err.append(row[-1])
        fig = plt.figure() 
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_rot, y_min_ax, z_err, label="Actual")
        ax.scatter(x_rot, y_min_ax, trained_model.predict(to_predict), label="Prediction")
        ax.legend()
        ax.set_title(f"Ellipses at location: ({loc_x}, {loc_y})")
        ax.grid()
        ax.set_xlabel("Rotation")
        ax.set_ylabel("Minor Axis size")
        ax.set_zlabel("MAE")
        plt.show()
        fig.savefig(f"BetaDataExper/HyperEllipsoid/closed_form_figs/polydeg{deg}/({loc_x},{loc_y})")


def main(datapath):
    df = pd.read_csv(datapath)
    array = df.to_numpy()
    X, y = array[:, :-1], array[:, -1]

    # lin = LinearRegression()
    # lin.fit(X, y)
    # y_pred = lin.predict(X)
    # print(f'MAE of polynomial degree 1: {mean_absolute_error(y, y_pred)}')

    # make_viz(array, 1, lin)

    for deg in range(1, 6):
        poly = PolynomialFeatures(degree = deg)
        X_poly = poly.fit_transform(X)
        
        poly.fit(X_poly, y)
        lin_poly = LinearRegression()
        lin_poly.fit(X_poly, y)

        y_pred = lin_poly.predict(X_poly)
        print(f'MAE of polynomial degree {deg}: {mean_absolute_error(y, y_pred)}')

        print(X_poly.shape, y.shape)
        new_array = np.append(X_poly, np.expand_dims(y, axis=1), axis=1)
        make_viz(new_array, deg, lin_poly)

# 4 3-D plots


if __name__ == "__main__":
    datapath = "BetaDataExper\HyperEllipsoid\pred_error.csv"
    main(datapath)
