import math

# Verify the "trick" where the percent difference (x−y/x)
# between two numbers close to 1 can be well approximated by the difference between the log of the two numbers (log(x)−log(y))

x = 1.05
y = 1.02


def pec_diff(x, y):
    return ((x-y) / x) * 100

print(pec_diff(x, y))


def log_xy(x, y):
    return (math.log(x) - math.log(y)) * 100

print(log_xy(x, y))


#replace

test = "abc"
print(test.replace("c", "d"))

#String formating

