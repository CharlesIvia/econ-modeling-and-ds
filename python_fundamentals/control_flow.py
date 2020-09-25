#Basic pricinciples of pricing assets with deterministic payoffs

#Two years => 1 / (1+r)**2

#Net present value

#Po = (1+r / r) - 1 / r (1 / 1+ r)**t


import datetime

halloween = datetime.date(2017, 10, 31)

print(halloween.month)

if halloween.month > 9:
    print("Halloween is in Q4")
elif halloween.month > 6:
    print("Halloween is in Q3")
elif halloween.month > 3:
    print("Halloween is in Q2")
else:
    print("Halloween is in Q1")

#iteration

for i in range(1, 11):
    print(f"{i}**2 = {i**2}")

# revenue by quarter
company_revenue = [5.12, 5.20, 5.50, 6.50]

for index, value in enumerate(company_revenue):
    print(f"quarter {index} revenue is ${value} million")

#accessing another vector.

cities = ["Phoenix", "Austin", "San Diego", "New York"]
states = ["Arizona", "Texas", "California", "New York"]
for index, city in enumerate(cities):
    state = states[index]
    print(f"{city} is in {state}")
    
    