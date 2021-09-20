import numpy as np

def get_position(z:np.ndarray, v:np.ndarray, i:int, t:int) -> np.ndarray:
    '''
    Calculates position of node i at time t.
    With the assumption of constant velocity of node i.

    :param z:   Latent node positions vector
    :param v:   Node velocities
    :param i:   Index of the node to get the position of
    :param t:   The current time

    :returns:   The current position of node i based on 
                its starting position and velocity
    '''
    return z[i, :] + v[i, :] * t

def contant_velocity(z0:np.ndarray, v0:np.ndarray, T:int, time_steps:int) -> list:
    '''
    Computes the latent node positions during a time interval 
    based on initial starting positions and contant node velocities.
    The node positions are computed for each step in the time interval
    which is based on the value of time_steps.
    The time interval i assumed to start from t = 0.

    :param z0:          List of initial node positions
    :param v0:          List of constant velocities to be used
                        throughout the time interval
    :param T:           The end of the time interval
    :param time_steps:  The number of steps to divide
                        the time interval from 0 to T
                        into.

    :returns:           A collection of lists. Each list in the
                        collection holds the node positions for
                        that given point in time.
    '''
    node_positions = [z0]
    for i, t in enumerate(np.linspace(0, T, time_steps)):
        z = node_positions[i]
        next_z = []
        for i, _ in enumerate(z):
            next_z.append(get_position(z, v0, i, t))
        node_positions.append(np.asarray(next_z))

    return node_positions