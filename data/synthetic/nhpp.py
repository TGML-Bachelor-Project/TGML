import numpy as np

class NHPP:
    def __init__(self, T, intensity_func, time_bins, seed=42) -> None:
        self.__T = T
        self.__intensity_func = intensity_func
        self.__time_bins = time_bins
        self.__seed = seed

        # Check time bin start and end
        if self.__time_bins[0] != 0 or self.__time_bins[-1] != self.__T:
            raise Exception('Invalid time intervals. Must start with t=0 and end with t=T')

        self.__numOfTimeBins = len(self.__timeBins)
        # Find the max lambda values for each interval, add [0] to start the indexing from 1
        self.__lambdaValues = [0] + [max(intensity_func(t=self.__time_bins[inx - 1]), intensity_func(t=self.__time_bins[inx])) 
                                        for inx in range(1, self.__numOfTimeBins)]

        # Set seed
        np.random.seed(self.__seed)

    def generate_time_units(self):
        '''
        Created based on the algorithm on page 86
        in chapter 5 of the book Generating Continuous Random Variables
        '''
        t, J, S = 0, 1, []

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

        return S


