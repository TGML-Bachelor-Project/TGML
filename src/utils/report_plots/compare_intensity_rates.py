import numpy as np
import matplotlib.pyplot as plt



def compare_intensity_rates_plot(train_t, result_model, gt_model, nodes):

    i = nodes[0]
    j = nodes[1]
    
    ## Compute learned as well as ground truth intensities for specific node pair
    result_train = []
    gt_train = []  
    for ti in train_t:
        result_train.append(result_model.log_intensity_function(i=i, j=j, t=ti))
        gt_train.append(gt_model.log_intensity_function(i=i, j=j, t=ti))

    ## Plot
    fig = plt.figure()
    ax1 = fig.add_subplot(111)    # The big subplot
    ax1.grid()
    ax1.plot(train_t, result_train , color="red", label="est.")
    ax1.plot(train_t, gt_train , color="blue", label="gt")
    ax1.legend()


    # Turn off axis lines and ticks of the big subplot
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Log Intensity")
    ax1.set_title(f"Interaction intensity betweeen node {i} and {j}")
    plt.show()
