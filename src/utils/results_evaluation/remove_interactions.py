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


def predict(model, sample_indices, t0, tn, triu_indices):
    scores = []
    for idx in sample_indices:
        u, v = triu_indices[0][idx], triu_indices[1][idx]
        score = model.riemann_sum(u, v, t0, tn, n_samples=10).item()
        scores.append(score)
    return np.array(scores)


def make_AUC_testset(num_nodes, removed_interactions):
    nodepair_ind = np.triu_indices(num_nodes, k=1)
    all_node_pairs = list(zip(nodepair_ind[0], nodepair_ind[1]))

    ## Sample alernative node pairs for each of the timestamps in the removed interactions
    alternate_node_pairs = []
    for tup in removed_interactions.tolist():
        tup_copy = tup
        excluded_node_pair = [tup[0], tup[1]]
        alternate_node_pair = random.choice([i for i in all_node_pairs if i not in excluded_node_pair])
        tup_copy[0], tup_copy[1] = alternate_node_pair[0], alternate_node_pair[1]
        alternate_node_pairs.append(tup_copy)

    ## Define positive and negative sets of node pairs
    pos_test_set, neg_test_set = removed_interactions.tolist(), alternate_node_pairs

    ## Define labels
    positive_labels = np.ones(shape=(len(removed_interactions.tolist())))
    negative_labels = np.zeros(shape=(len(removed_interactions.tolist())))
    labels = np.concatenate([positive_labels, negative_labels], axis=0)

    return pos_test_set, neg_test_set, labels

    

def auc_removed_interactions(removed_interactions, num_nodes, result_model, wandb_handler, gt_model=None):
    
    if gt_model is None:
        pos_test_set, neg_test_set, labels = make_AUC_testset(num_nodes=num_nodes, removed_interactions=removed_interactions)

        ## Compute probability for node pair interaction for test set
        pos_probs, neg_probs = [], []
        for tup in pos_test_set:
            pos_probs.append(float(result_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
        for tup in neg_test_set:
            neg_probs.append(float(result_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
        probs = np.concatenate([pos_probs, neg_probs], axis=0)

        ## Compute ROC metrics
        fpr, tpr, thresh = roc_curve(labels, probs, pos_label=1)
        auc_score = roc_auc_score(labels, probs)
        wandb_handler.log({'false_positive_rate':fpr, 'true_poitive_rate':tpr, 'thresh':thresh, 'AUC_score': auc_score})

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

        return fpr, tpr, thresh, auc_score
        
    else:
        pos_test_set, neg_test_set, labels = make_AUC_testset(num_nodes=num_nodes, removed_interactions=removed_interactions)

        ## Compute probability for node pair interaction for test set
        pos_probs, neg_probs = [], []
        gt_pos_probs, gt_neg_probs = [], []
        
        for tup in pos_test_set:
            pos_probs.append(float(result_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
            gt_pos_probs.append(float(gt_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
        for tup in neg_test_set:
            neg_probs.append(float(result_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
            gt_neg_probs.append(float(gt_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
        
        probs = np.concatenate([pos_probs, neg_probs], axis=0)
        gt_probs = np.concatenate([gt_pos_probs, gt_neg_probs], axis=0)


        ## Compute ROC metrics
        fpr, tpr, thresh = roc_curve(labels, probs, pos_label=1)
        auc_score = roc_auc_score(labels, probs)
        wandb_handler.log({'false_positive_rate':fpr, 'true_poitive_rate':tpr, 'thresh':thresh, 'AUC_score': auc_score})

        gt_fpr, gt_tpr, gt_thresh = roc_curve(labels, gt_probs, pos_label=1)
        gt_auc_score = roc_auc_score(labels, gt_probs)
        wandb_handler.log({'GT_false_positive_rate':gt_fpr, 'GT_true_poitive_rate':gt_tpr, 'GT_thresh':gt_thresh, 'GT_AUC_score': gt_auc_score})

        ## Plot ROC Curve
        fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
        plt.style.use('seaborn')

        # plot roc curves
        ax.plot(fpr, tpr, linestyle='--',color='red', label='est. Model')
        ax.plot(gt_fpr, gt_tpr, linestyle='--',color='blue', label='gt. Model')
        random_probs = [0 for i in range(len(labels))]
        p_fpr, p_tpr, _ = roc_curve(labels, random_probs, pos_label=1)
        ax.plot(p_fpr, p_tpr, linestyle='--', color='black')
        
        ax.grid()
        plt.title('ROC curve')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive rate')

        ax.legend(loc='best')

        wandb_handler.log({'ROC_curve': wandb_handler.Image(fig)})

        return fpr, tpr, thresh, auc_score
















        # nodepair_ind = np.triu_indices(num_nodes, k=1)
        # all_node_pairs = list(zip(nodepair_ind[0], nodepair_ind[1]))

        # ## First we sample alernative node pairs for each of the timestamps in the removed interactions
        # alternate_node_pairs = []
        # for tup in removed_interactions.tolist():
        #     tup_copy = tup
        #     excluded_node_pair = [tup[0], tup[1]]
        #     alternate_node_pair = random.choice([i for i in all_node_pairs if i not in excluded_node_pair])
        #     tup_copy[0], tup_copy[1] = alternate_node_pair[0], alternate_node_pair[1]
        #     alternate_node_pairs.append(tup_copy)

        # ## Then we create a new dataset with labels which can be evaluated
        # test_set = []
        # labels = []
        # for i in range(len(removed_interactions.tolist())):
        #     if random.randint(0,1) == 0:
        #         test_set.append(alternate_node_pairs[i])
        #         labels.append(0)
        #     else:
        #         test_set.append(removed_interactions[i].tolist())
        #         labels.append(1)

        # ## Compute probability for node pair interaction for test set
        # probs = []
        # for tup in test_set:
        #     probs.append(float(result_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))
        # gt_probs = []
        # for tup in test_set:
        #     gt_probs.append(float(gt_model.log_intensity_function(i=int(tup[0]), j=int(tup[1]), t=tup[2])))

        # ## Compute ROC metrics
        # fpr, tpr, thresh = roc_curve(labels, probs, pos_label=1)
        # auc_score = roc_auc_score(labels, probs)
        # wandb_handler.log({'false_positive_rate':fpr, 'true_poitive_rate':tpr, 'thresh':thresh, 'AUC_score': auc_score})

        # gt_fpr, gt_tpr, gt_thresh = roc_curve(labels, gt_probs, pos_label=1)
        # gt_auc_score = roc_auc_score(labels, gt_probs)
        # wandb_handler.log({'GT_false_positive_rate':gt_fpr, 'GT_true_poitive_rate':gt_tpr, 'GT_thresh':gt_thresh, 'GT_AUC_score': gt_auc_score})

        # ## Plot ROC Curve
        # fig, ax = plt.subplots(1,1, figsize=(10, 6), facecolor='w', edgecolor='k')
        # plt.style.use('seaborn')
        # # plot roc curves
        # ax.plot(fpr, tpr, linestyle='--',color='red', label='est. Model')
        # ax.plot(gt_fpr, gt_tpr, linestyle='--',color='blue', label='gt. Model')
        # random_probs = [0 for i in range(len(labels))]
        # p_fpr, p_tpr, _ = roc_curve(labels, random_probs, pos_label=1)
        # ax.plot(p_fpr, p_tpr, linestyle='--', color='black')
        
        # ax.grid()
        # plt.title('ROC curve')
        # ax.set_xlabel('False Positive Rate')
        # ax.set_ylabel('True Positive rate')

        # ax.legend(loc='best')


        # wandb_handler.log({'ROC_curve': wandb_handler.Image(fig)})
        #plt.show()