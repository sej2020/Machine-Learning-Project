import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import time

# ["location_x", "location_y", "rotation", "major_axis_length", "minor_axis_length", "MAE"]

def make_viz(array, deg, trained_model, dummys, metric):

    for (loc_x, loc_y), dummy in zip([(0,0), (0,10), (10,0), (10,10)], dummys):
        x_rot = []
        y_min_ax = []
        z_err = []
        # to_predict = []
        for row in array:
            if row[1] == loc_x and row[2] == loc_y:
            #     to_predict.append(row[:-1])
                x_rot.append(row[3])
                y_min_ax.append(row[5])
                z_err.append(row[-1])
        fig = plt.figure() 
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_rot, y_min_ax, z_err, label="Actual")
        ax.scatter(dummy[:,3], dummy[:,5], trained_model.predict(dummy), label="Prediction", marker="_")
        ax.legend()
        ax.set_title(f"Ellipses at location: ({loc_x}, {loc_y})\nPolynomial of degree {deg}")
        ax.grid()
        ax.set_xlabel("Rotation")
        ax.set_ylabel("Minor Axis size")
        ax.set_zlim(min(array[:,-1])*1.1,max(array[:,-1])*1.1)
        ax.set_zlabel(metric)
        # plt.show()
        # fig.savefig(f"BetaDataExper/HyperEllipsoid/closed_form_figs/{metric}/polydeg{deg}/({loc_x},{loc_y})")


def main(datapath, metric):
    df = pd.read_csv(datapath)
    array = df.to_numpy()
    X, y = array[:, :-1], array[:, -1]

    # lin = LinearRegression()
    # lin.fit(X, y)
    # y_pred = lin.predict(X)
    # print(f'MAE of polynomial degree 1: {mean_absolute_error(y, y_pred)}')

    # make_viz(array, 1, lin)

    for deg in range(1, 6):

        dummy_Xs = interp_data(deg)

        poly = PolynomialFeatures(degree = deg)
        X_poly = poly.fit_transform(X)

        poly.fit(X_poly, y)
        lin_poly = LinearRegression()
        lin_poly.fit(X_poly, y)

        y_pred = lin_poly.predict(X_poly)
        print(f'MAPE of polynomial degree {deg} for {metric} prediction: {round(mean_absolute_percentage_error(y, y_pred),4)}')

        new_array = np.append(X_poly, np.expand_dims(y, axis=1), axis=1)
        make_viz(new_array, deg, lin_poly, dummy_Xs, metric)

# 4 3-D plots

# ["location_x", "location_y", "rotation", "major_axis_length", "minor_axis_length", "MAE"]

def interp_data(deg):
    dummy = PolynomialFeatures(degree = deg)
    ph00 = np.zeros(shape=(10000,5))
    ph010 = np.zeros(shape=(10000,5))
    ph100 = np.zeros(shape=(10000,5))
    ph1010 = np.zeros(shape=(10000,5))

    ph010[:, 1] = np.array([10 for _ in range(10000)])
    ph100[:, 0] = np.array([10 for _ in range(10000)])
    ph1010[:, 0] = np.array([10 for _ in range(10000)])
    ph1010[:, 1] = np.array([10 for _ in range(10000)])

    dummy_Xs = []
    for ph in [ph00, ph010, ph100, ph1010]:
        ph[:, 2] = np.array([i/100 for i in range(0, 90*100, 90) for j in range(100)])
        ph[:, 3] = np.array([10 for _ in range(10000)])
        ph[:, 4] = np.array([i/100 for j in range(100) for i in range(1*100, 10*100, 9)])
        X_dummy = dummy.fit_transform(ph)
        dummy_Xs.append(X_dummy)
    
    return dummy_Xs


if __name__ == "__main__":
    
    for metric in ["MAE", "MSE", "RMSE", "R2", "elapsed_time"]:
        datapath = f"BetaDataExper/HyperEllipsoid/closed_form_figs/pred_error_{metric}.csv"
        main(datapath, metric)
        ## thread metric thru in place of MAE
