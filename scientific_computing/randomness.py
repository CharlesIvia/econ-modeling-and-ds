import numpy as np
import matplotlib.pyplot as plt
import timeit

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

# plt.show()

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


# A faster alternative using numpy array


def simulate_loan_repayments(N, r=0.05, repayment_full=25000, repayment_part=12500):
    """Simulate present value of N loans given values for discount rate and
    repayment values
    """
    random_numbers = np.random.rand(N)

    # start as 0 -- no repayment

    repayment_sims = np.zeros(N)

    # adjust for full and partial repayment

    partial = random_numbers <= 0.20
    repayment_sims[partial] = repayment_part

    full = ~partial & (random_numbers <= 0.95)
    repayment_sims[full] = repayment_full

    repayment_sims = (1 / (1 + r)) * repayment_sims

    return repayment_sims


print(np.mean(simulate_loan_repayments(25000)))

# print(timeit.timeit(lambda: np.mean(simulate_loan_repayments_slow(25000))))
# print(timeit.timeit(lambda: np.mean(simulate_loan_repayments(25000))))

print(timeit.timeit(f"{np.mean(simulate_loan_repayments_slow(25000))}"))
print(timeit.timeit(f"{np.mean(simulate_loan_repayments(25000))}"))


# Profitability threshold

# Finding the largest loan size that ensures we get 95% probability of profitability
# in a year we make 250 loans


def simulate_year_of_loans(N=250, K=1000):
    # Create an array to store the values
    avg_repayments = np.zeros(K)

    for year in range(K):
        repaid_year = 0.0
        n_loans = simulate_loan_repayments(N)
        avg_repayments[year] = n_loans.mean()

    return avg_repayments


loan_repayment_outcomes = simulate_year_of_loans(N=250)
lro_5 = np.percentile(loan_repayment_outcomes, 5)
print(lro_5)

# Markov chain in simulating loan repayment status


def simulate_loan_lifetime(monthly_payment):

    # Create arrays to store outputs
    payments = np.zeros(12)
    # Note: dtype 'U12' means a string with no more than 12 characters
    statuses = np.array(
        4 * ["repaying", "delinquency", "default"], dtype="U12")

    # Everyone is repaying during their first month
    payments[0] = monthly_payment
    statuses[0] = "repaying"

    for month in range(1, 12):
        rn = np.random.rand()

        if statuses[month - 1] == "repaying":
            if rn < 0.85:
                payments[month] = monthly_payment
                statuses[month] = "repaying"
            elif rn < 0.95:
                payments[month] = 0.0
                statuses[month] = "delinquency"
            else:
                payments[month] = 0.0
                statuses[month] = "default"
        elif statuses[month - 1] == "delinquency":
            if rn < 0.25:
                payments[month] = monthly_payment
                statuses[month] = "repaying"
            elif rn < 0.85:
                payments[month] = 0.0
                statuses[month] = "delinquency"
            else:
                payments[month] = 0.0
                statuses[month] = "default"
        else:  # Default -- Stays in default after it gets there
            payments[month] = 0.0
            statuses[month] = "default"

    return payments, statuses
