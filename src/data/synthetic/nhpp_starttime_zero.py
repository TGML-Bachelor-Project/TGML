import numpy as np

class NHPP:
    '''
    Class for simulating Non-homogeneously distributed time points
    for node pair interactions based on a given intensity function.

    '''
    def __init__(self, max_time:int, intensity_func, time_bins:list, seed:int=0, t_start:int=0) -> None:
        '''
        :param T:               The end time point of the simulation
        :param intensity_func:  A given function defining how the intensity between node pairs 
                                is computed
        :param time_bins:       A collection of time points for splitting the time interval of
                                the simulation into discrete time bins
        :param seed:            The random seed for the simulation
        '''
        self.__t_start_true = t_start
        self.__t_start = t_start - t_start
        self.__max_time = max_time - t_start
        self.__intensity_func = intensity_func
        self.__time_bins = time_bins - t_start
        self.__seed = seed

        self.__numOfTimeBins = len(self.__time_bins)
        # Find the max lambda values for each interval, add [0] to start the indexing from 1
        self.__lambdaValues = [0] + [max(intensity_func(t=self.__time_bins[inx - 1]), 
                                        intensity_func(t=self.__time_bins[inx])) 
                                        for inx in range(1, self.__numOfTimeBins)]

        # Set seed
        np.random.seed(self.__seed)

    def generate_time_units(self) -> list:
        '''
        Created based on the algorithm on page 86
        in chapter 5 of the book Generating Continuous Random Variables

        The function generates Non-homogeneously distributed time points for 
        node interactions.

        :param t:   The start time to simulate node interactions from

        :returns:   A list of floating point numbers representing the time points
                    of interaction for a given pair of nodes based on the given 
                    intensity function
        '''
        t, J, S = self.__t_start, 1, []

        # Step 2
        U = np.random.uniform(low=0, high=1) # Random number
        X = (-1/self.__lambdaValues[J]) * np.log(U) # Random variable from exponential dist for NHPP time step

        while True:
            # Step 3
            if t + X < self.__time_bins[J]:
                # Step 4
                t = t + X
                # Step 5
                U = np.random.uniform(low=0, high=1)
                # Step 6
                if U <= self.__intensity_func(t)/self.__lambdaValues[J]:
                    #Don't need I for index, because we append t to S
                    S.append(t)
                # Step 7 -> Do step 2 then loop starts again at step 3
                U = np.random.uniform(low=0, high=1) # Random number
                X = (-1/self.__lambdaValues[J]) * np.log(U) # Random variable from exponential dist for NHPP time step
            else:
                # Step 8
                if J == self.__numOfTimeBins - 1: #k +1 because zero-indexing
                    break
                # Step 9
                X = (X-self.__time_bins[J] + t) * self.__lambdaValues[J]/self.__lambdaValues[J+1]
                t = self.__time_bins[J]
                J += 1
                # Step 10 -> Loop starts over going back to step 3

        return S + self.__t_start_true