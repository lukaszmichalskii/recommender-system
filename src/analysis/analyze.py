import matplotlib

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

matplotlib.use("agg")  # make backend thread safe


def analyze(mdeval_data, history, output):
    plot = cf_learn_plot(history)
    save_plot(plot, output.joinpath("cf_learn.png"))

    plot = model_evaluation(mdeval_data)
    save_plot(plot, output.joinpath("model_evaluation.png"))


def cf_learn_plot(history):
    iter, loss = [], []
    for x, y in history:
        iter.append(x)
        loss.append(y)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(iter, loss, marker="o", c="red")
    ax.plot(iter, loss, c="orange")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.set_title("Model loss over time")
    return fig


def model_evaluation(evaluation):
    original, predicted = [], []
    for y1, y2 in evaluation:
        original.append(y1)
        predicted.append(y2)
    x = [i for i in range(len(original))]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.scatter(x, predicted, marker="o", c="red")
    ax.plot(x, predicted, c="red", label="predictions")
    ax.scatter(x, original, marker="o", c="green")
    ax.plot(x, original, c="green", label="original")
    ax.set_title(
        f"Model predictions evaluation. MSE = {mean_squared_error(original, predicted):.2f}"
    )
    ax.grid(True)
    ax.legend()
    ax.set_ylabel("Rating [1-5]")
    ax.set_xlabel(f"Rating number")
    return fig


def save_plot(plot, path):
    plot.savefig(path)
