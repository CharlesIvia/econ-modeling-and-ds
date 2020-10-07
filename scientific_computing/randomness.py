import numpy as np
import matplotlib.pyplot as plt

# activate plot theme
import qeds

qeds.themes.mpl_style()

# Probability- discrete (countable evt space) and continous(non-countable) random variable

# Simulating randomness in Python


print(np.random.rand())


# array of random numbers

arr_random = np.random.rand(25)
print(arr_random)

rand_mat = np.random.rand(5, 5)
print(rand_mat)

rand_mat_two = np.random.rand(2, 3, 4)
print(rand_mat_two)
