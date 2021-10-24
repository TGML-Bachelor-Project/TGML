import numpy as np
import matplotlib.pyplot as plt


def compare_intensity_rates_plot(train_t, test_t, result_model, gt_model, nodes):

    i = nodes[0]
    j = nodes[1]
    plot_t = [train_t, test_t]
    
    ## Compute learned as well as ground truth intensities for specific node pair
    result_train = []
    gt_train = []  
    for ti in plot_t[0]:
        result_train.append(result_model.log_intensity_function(i=i, j=j, t=ti))
        gt_train.append(gt_model.log_intensity_function(i=i, j=j, t=ti))
    
    result_test = []
    gt_test = []
    for ti in plot_t[1]:
        result_test.append(result_model.log_intensity_function(i=i, j=j, t=ti))
        gt_test.append(gt_model.log_intensity_function(i=i, j=j, t=ti))

    ## Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)    # The big subplot
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)#, sharex=ax1, sharey=ax1)
    ax1.grid()
    ax2.grid()
    ax1.plot(plot_t[0], result_train , color="red", label="est.")
    ax1.plot(plot_t[0], gt_train , color="blue", label="gt")
    ax1.legend()
    ax1.set_title("Train")
    ax2.plot(plot_t[1], result_test , color="red", label="est.")
    ax2.plot(plot_t[1], gt_test , color="blue", label="gt")
    ax2.legend()
    ax2.set_title("Test")

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("Time")
    ax.set_ylabel("Log Intensity")
    ax.set_title(f"Interaction intensity betweeen node {i} and {j}")
    plt.show()