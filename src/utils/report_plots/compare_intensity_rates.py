import numpy as np
import matplotlib.pyplot as plt
import time


def compare_intensity_rates_plot(train_t, result_model, gt_model, nodes, wandb_handler):


    res_list = []
    gt_list = []
    for node_pair in nodes:
        i = node_pair[0]
        j = node_pair[1]
        
        ## Compute learned as well as ground truth intensities for specific node pair
        res = []
        gt = []  
        for ti in train_t:
            res.append(result_model.log_intensity_function(i=i, j=j, t=ti))
            gt.append(gt_model.log_intensity_function(i=i, j=j, t=ti))
        
        res_list.append(res)
        gt_list.append(gt)


    ## Plot
    fig, axs = plt.subplots(1,len(res_list), figsize=(5*len(nodes), 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.5)

    axs = axs.ravel()

    for i in range(len(res_list)):
        axs[i].grid()
        axs[i].plot(train_t, res_list[i], color="red", label="est.")
        axs[i].plot(train_t, gt_list[i] , color="blue", label="gt")
        axs[i].legend()
        axs[i].set_title(f"Interactions Intensity Node {nodes[i][0]} and {nodes[i][1]}")


    wandb_handler.log({'intensity_plot': wandb_handler.Image(fig)})
    # plt.show()
    return 
    