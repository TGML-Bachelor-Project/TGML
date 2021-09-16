import numpy as np

from utils.visualize.positions import node_positions

def get_position(z0, v0, i:int, t:int) -> np.ndarray:
    '''
    Calculates position of node i at time t.
    With the assumption of constant velocity of node i.

    :param i:   Index of the node to get the position of
    :param t:   The current time

    :returns:   The current position of node i based on 
                its starting position and velocity
    '''
    return z0[i, :] + v0[i, :] * t

def compute_node_positions(z0:list, v0:list, T:int, time_steps:int) -> list:
    node_positions = []
    for t in np.linspace(0, T, time_steps):
