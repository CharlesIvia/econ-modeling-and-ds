import datetime
import numpy as np

#Zero-coupon bonds

paid = 100
time = 10
interest = 0.05


def zero_coupon_bond(p, t, r):
    return p / ((1 + r)**t)

print(zero_coupon_bond(paid, time, interest))


#Greetings

current_time = datetime.datetime.now()

hour = current_time.hour

if hour > 6 and hour < 12:
    print("Good morning!")

elif hour > 12 and hour < 18:
    print("Good evening!")


#Working with random numbers

x = np.random.random()

print(x)

if x < 0.5:
    print(f"{x} is less than 0.5")
else:
    print(f"{x} is greater than 0.5")


#Work-education decision

# Discount rate
r = 0.05

# High school wage
w_hs = 40000

# College wage and cost of college
c_college = 5000
w_college = 50000


def npv(paid, time, interest):
    P0 = paid / ((1+interest)**time)
    return P0

# Compute npv of being a hs worker

hs_npv = npv(w_hs, 40, r)
print(hs_npv)

# Compute npv of attending college

college_cost_npv = npv(c_college, 4, r)
print(college_cost_npv)

# Compute npv of being a college worker

c_npv = npv(w_college, 36, r)
print(c_npv)

# Is npv_collegeworker - npv_collegecost > npv_hsworker

if c_npv - college_cost_npv > hs_npv:
    print("The student should got to college")
else:
    print("Student not got to college")