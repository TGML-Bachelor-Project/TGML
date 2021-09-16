import matplotlib.pyplot as plt

def logloss(loglosses:dict) -> None:
    '''
    Plots the logloss for each entry in the 
    given dictionary of loglosses.

    :param loglosses:   A dictionary with key = name of logloss, and
                        val = list of loglosses for each epoch
    '''
    plt.title('Model Log Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')

    for k in loglosses.keys():
        plt.plot(loglosses[k], label=k)

    plt.legend()
    plt.show()