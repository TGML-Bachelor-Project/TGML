import random
random.seed(1)
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import time

def remove_interactions(dataset, percentage, device):

    dataset_reduced = []
    removed_interactions = []
    for tup in dataset.tolist():
        if random.random() > percentage:
            dataset_reduced.append(tup)
        else:
            removed_interactions.append(tup)

    dataset_reduced = torch.from_numpy(np.asarray(dataset_reduced)).to(device)
    removed_interactions = torch.from_numpy(np.asarray(removed_interactions)).to(device)
    print(f'Reduced training set by random selection, interactions: {len(dataset_reduced)}')
    print(f'Removed number of interactions: {len(removed_interactions)}')
    return dataset_reduced, removed_interactions


def auc_removed_interactions(removed_interactions, num_nodes, result_model, wandb_handler):
    
    nodepair_ind = np.triu_indices(num_nodes, k=1)
    all_node_pairs = list(zip(nodepair_ind[0], nodepair_ind[1]))

    ## First we sample alernative node pairs for each of the timestamps in the removed interactions
    alternate_node_pairs = []
    for tup in removed_interactions.tolist():
        tup_copy = tup
        excluded_node_pair = [tup[0], tup[1]]
        alternate_node_pair = random.choice([i for i in all_node_pairs if i not in excluded_node_pair])
        tup_copy[0], tup_copy[1] = alternate_node_pair[0], alternate_node_pair[1]
        alternate_node_pairs.append(tup_copy)

    ## Then we create a new dataset with labels which can be evaluated
    test_set = []
    labels = []
    for i in range(len(removed_interactions.tolist())):
        if random.randint(0,1) == 0:
            test_set.append(alternate_node_pairs[i])
            labels.append(0)
        else:
            test_set.append(removed_interactions[i].tolist())
            labels.append(1)

    ## Compute probability for node pair interaction for test set
    probs = []
    for tup in test_set:
        probs.append(float(result_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))

    ## Compute ROC metrics
    fpr, tpr, thresh = roc_curve(labels, probs, pos_label=1)
    auc_score = roc_auc_score(labels, probs)


    ## Plot ROC Curve
    fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
    plt.style.use('seaborn')
    # plot roc curves
    ax.plot(fpr, tpr, linestyle='--',color='orange', label='Constant Velocity Model')
    random_probs = [0 for i in range(len(labels))]
    p_fpr, p_tpr, _ = roc_curve(labels, random_probs, pos_label=1)
    ax.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    
    ax.grid()
    plt.title('ROC curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive rate')

    ax.legend(loc='best')


    wandb_handler.log({'ROC_curve': wandb_handler.Image(fig)})
    plt.show()

    return fpr, tpr, thresh, auc_score