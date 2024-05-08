import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from darts import TimeSeries
from darts.models import NaiveDrift, NaiveSeasonal, ExponentialSmoothing, Theta
from darts.utils.statistics import check_seasonality
from darts.metrics.metrics import mape

def main():
    #import data
    data = pd.read_csv("AirPassengers.csv")

    # visualize data
    plt.plot(data["Month"], data["#Passengers"])
    plt.xlabel("Month")
    plt.ylabel("Number of Passengers")
    plt.title("Number of Passengers Over Time")
    plt.show()

    # convert into timeseries
    data_timeseries = TimeSeries.from_dataframe(data, "Month", "#Passengers")
    train, val = data_timeseries[:-36], data_timeseries[-36:]

    train.plot()
    #plt.legend()
    #plt.show()

    # create and fit model
    model = NaiveDrift()
    model.fit(train)
    prediction = model.predict(len(val))
    prediction.plot(label="Prediction")
    val.plot(label="Actual")
    plt.legend()
    plt.show()

    # check seasonality
    is_seasonal = check_seasonality(data_timeseries)
    print(is_seasonal)
    if is_seasonal[0]:
        model_seasonal = NaiveSeasonal(K=is_seasonal[1])
        model_seasonal.fit(train)
        prediction_seasonal = model_seasonal.predict(len(val))
        train.plot()
        prediction_seasonal.plot(label="Prediction")
        val.plot(label="Actual")
        plt.legend()
        plt.show()

        # calculate error
        error = mape(val, prediction_seasonal)
        print(error)

    # create and fit exponential smoothing model
    model_exp = ExponentialSmoothing()
    model_exp.fit(train)
    prediction_exp = model_exp.predict(len(val))
    train.plot()
    prediction_exp.plot(label="Prediction")
    val.plot(label="Actual")
    plt.legend()
    plt.show()

    # create and fit theta model
    model_theta = Theta()
    model_theta.fit(train)
    prediction_theta = model_theta.predict(len(val))
    train.plot()
    prediction_theta.plot(label="Prediction")
    val.plot(label="Actual")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    main()