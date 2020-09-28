#Zero-coupon bonds

paid = 100
time = 10
interest = 0.05


def zero_coupon_bond(p, t, r):
    return p / ((1 + r)**t)

print(zero_coupon_bond(paid, time, interest))
