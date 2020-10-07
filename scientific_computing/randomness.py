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


# Need for randomness

# Demo law of large numbers by approximating the uniform distribution

# Draw various numbers of uniform[0, 1] random variables
draws_10 = np.random.rand(10)
draws_200 = np.random.rand(200)
draws_10000 = np.random.rand(10000)

# Plot their histograms

fig, ax = plt.subplots(3)

ax[0].set_title("Histogram with 10 draws")
ax[0].hist(draws_10)

ax[1].set_title("Histogram with 200 draws")
ax[1].hist(draws_200)

ax[2].set_title("Histogram with 10,000 draws")
ax[2].hist(draws_10000)

fig.tight_layout()

plt.show()

# Discrete distribution


def simulate_loan_repayments_slow(
    N, r=0.05, repayment_full=25000, repayment_part=12500
):
    repayment_sims = np.zeros(N)
    for i in range(N):
        x = np.random.rand()  # Draw a random number

        # Fu;ll repayment 75% of time

        if x < 0.75:
            repaid = repayment_full
        elif x < 0.95:
            repaid = repayment_part
        else:
            repaid = 0.0

        repayment_sims[i] = (1 / (1 + r)) * repaid
    return repayment_sims


print(np.mean(simulate_loan_repayments_slow(25000)))
