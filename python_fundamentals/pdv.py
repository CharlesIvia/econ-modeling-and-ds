#compute the present discounted value of a payment (D) made in T years assuming an interest rate of 2.5%Let's compute the present discounted value of a payment (D) made in T years assuming an interest rate of 2.5

#Formula  PDV = D / (1+R)T


def pdv(d, r, t):
    return d / (1+r)**t

D = 10000
R = 2.5/100
T = 5

print(pdv(D, R, T))