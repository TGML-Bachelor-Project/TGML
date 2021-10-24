import numpy as np
import matplotlib.pyplot as plt


def compare_intensity_rates_plot(train_t, test_t, result_model, gt_model):

    plot_t = [train_t, test_t]
    result_train = []
    gt_train = []
    
    print("Plot train")
    for ti in plot_t[0]:
        result_train.append(result_model.log_intensity_function(i=0, j=1, t=ti))
        gt_train.append(gt_model.log_intensity_function(i=0, j=1, t=ti))
    
    result_test = []
    gt_test = []
    print("Plot test")
    for ti in plot_t[1]:
        result_test.append(result_model.log_intensity_function(i=0, j=1, t=ti))
        gt_test.append(gt_model.log_intensity_function(i=0, j=1, t=ti))

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(plot_t[0], result_train, color="red", label="est")
    ax[0].plot(plot_t[0], gt_train, color="blue", label="gt")
    ax[0].legend()
    ax[0].set_title("Train")
    ax[1].plot(plot_t[1], result_test, color="red", label="est")
    ax[1].plot(plot_t[1],gt_test, color="blue", label="gt")
    ax[1].legend()
    ax[1].set_title("Test")
    plt.show()