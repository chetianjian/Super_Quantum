import numpy as np
import pandas as pd
import plotly
from sklearn.linear_model import LinearRegression


def seriesPosNegSumRatio(series):
    """
    :param series: 一列数据
    :return: 将该列数据所有正数与负数分别求和，再求比值: 正数之和 / 绝对值(负数之和)
    """

    positive_sum, negative_sum = np.nansum(series[series >= 0]), np.nansum(series[series < 0])
    return np.nan if negative_sum == 0 else positive_sum / abs(negative_sum)


def arrAvgAbs(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: 计算数组的平均绝对偏差。偏差表示每个数值与平均值之间的差，平均偏差表示每个偏差绝对值的平均值。
    """
    result = np.nanmean(abs(arr - np.nanmean(arr)))
    return result if not fillna else fillna


def rowWeighted(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: 各自按自身占总体的比例加权
    """
    result = arr / np.nansum(arr)
    return result if not fillna else result.fillna(fillna)


def arrNormalize(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: 令行向量模长 = 1，即 L^2 范数为 1。
    """
    result = arr / np.nansum(arr ** 2)
    return result if not fillna else result.fillna(fillna)


def arrStandardize(arr, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param arr: 一列数据
    :return: (arr - mean(arr)) / std(arr)
    """
    result = (arr - arr.mean()) / arr.std()
    return result if not fillna else result.fillna(fillna)


def seriesStandardize(series, fillna=False):
    """
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :param series: 一列数据
    :return: (series - mean(series)) / std(series)
    """
    if type(series) == pd.DataFrame:
        series = series.iloc[:, 0]
    result = (series - np.nanmean(series)) / np.nanstd(series)
    return result if not fillna else result.fillna(fillna)


def mvNeutralize(df: pd.DataFrame, mv: pd.DataFrame, fillna=False) -> pd.DataFrame:
    """
    :param df: Objective DataFrame which is going to be Neutralized.
    :param mv: DataFrame which records MV data.
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :return: Factor DataFrame neutralized cross-sectionally by Market Value.
    """
    assert df.shape == mv.shape
    data, mv = df.fillna(0), mv.fillna(0)
    for row in range(len(data)):
        features = data.iloc[row, :].values.reshape(-1, 1)
        pred = LinearRegression().fit(X=features, y=mv.iloc[row, :].values).predict(features)
        data.iloc[row, :] = data.iloc[row, :] - pred

    return data if not fillna else data.fillna(fillna)


def EMA(df: pd.DataFrame, window, fillna=False) -> pd.DataFrame:
    """
    :param df: Objective DataFrame.
    :param window: Use to compute alpha: alpha=2 / (window+1)
    :param fillna: If fill NaNs, default for False, otherwise input a value.
    :return: Exponential Moving Average.
    """

    return df.ewm(alpha=2/(window+1)).mean() if not fillna else df.ewm(alpha=2/(window+1)).mean().fillna(fillna)


def draw_line(series, jupyter=True, color="blue", description=None):
    if type(series) == pd.DataFrame:
        series = series.iloc[:, 0]

    traces = []
    trace = plotly.graph_objs.Scattergl(
        x=series.index,
        y=series.values,
        line=dict(color=color)
    )

    traces.append(trace)

    if description:
        layout = plotly.graph_objs.Layout(
            title=description
        )
    else:
        layout = plotly.graph_objs.Layout(
            title="Plot series data of: " + series.name
        )

    fig = plotly.graph_objs.Figure(data=traces, layout=layout)
    if jupyter:
        plotly.offline.init_notebook_mode(connected=True)
    return plotly.offline.iplot(fig, filename="dataplot")


def impluse(arr):
    result, i = 0, 0
    while i < len(arr):
        if np.isnan(arr[i]):
            i += 1
            continue
        direction = arr[i] > 0
        j = i
        while j + 1 < len(arr) and (arr[j + 1] > 0) == direction and not np.isnan(arr[j + 1]):
            j += 1
        if direction:
            result += (j - i + 1) ** 2
        else:
            result -= (j - i + 1) ** 2
        i = j + 1
    return result


def grad_desc_geometry(w, accuracy=1e-7, max_iter=10000000, step=5e-6):
    common_ratio = 1 / w ** 2
    result = common_ratio ** (w - 1) - w * common_ratio + w - 1
    for i in range(max_iter):
        gradient = (w - 1) * common_ratio ** (w - 2) - w + 1
        common_ratio -= step * gradient
        new_result = common_ratio ** (w - 1) - w * common_ratio + w - 1
        diff = abs(new_result - result)

        if i % 1000000 == 0:
            print(f"Completed {i} iterations, minimum approximates: {common_ratio}, accuracy: {diff}")
            if diff < accuracy:
                print(f"Accuracy 1e-7 triggered, minimum approximates: {common_ratio}, accuracy: {diff}")
                print("Algorithm Terminated.")
                break
        result = new_result
    return common_ratio


def dfReLU(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only non-positive values in a DataFrame, and replace all other values by 0.
    return df[df >= 0].fillna(0)


def dfRemoveInf(df: pd.DataFrame, fillna=False) -> pd.DataFrame:
    """
    :param df: Objective DataFrame.
    :param fillna: Whether to keep NaNs, default is False, otherwise input a value.
    :return: Remove all infinite values within a DataFrame.
    """

    return df[~np.isinf(df)] if not fillna else df[~np.isinf(df)].fillna(fillna)


def jackknife(series: np.array, method) -> tuple:
    """
    :param series: An array.
    :param method: Type of estimation: str in ["variation", "variance", "mean"]
    :return: (Jackknife Estimation, Jackknife Bias, Jackknife Variance)
    """

    if not series.sum():
        return np.nan, np.nan, np.nan
    theta_j_lst, stack = [], 0

    if method == "variation":
        estimator = lambda arr: np.nanstd(arr) / np.nanmean(arr)
    elif method == "variance":
        estimator = lambda arr: np.nanvar(arr)
    elif method == "mean":
        estimator = lambda arr: np.nanmean(arr)

    array = series.values

    for j in range(len(array)):
        leaved = np.delete(array, j)
        theta_j_lst.append(estimator(leaved))

    estimate = estimator(array)
    Jack_Bias = (len(array) - 1) * (np.nansum(theta_j_lst) / len(array) - estimate)

    for j in range(len(array)):
        stack += (theta_j_lst[j] - np.nansum(theta_j_lst) / len(array)) ** 2
    Jack_Var = (1 - 1 / len(array)) * stack

    return estimate - Jack_Bias, Jack_Bias, Jack_Var


def CP(series: pd.Series):
    series = (series.diff(1) / series.shift(1)).reset_index(drop=True)
    if not series.any():
        return np.nan
    avg, std = series.mean(), series.std()
    upper = series[series > avg + std]
    lower = series[series < avg - std]
    if len(upper) == 0:
        upper = 0
    else:
        upper = np.nanmedian(upper.index)
    if len(lower) == 0:
        lower = 0
    else:
        lower = np.nanmedian(lower.index)
    return lower - upper
