import numpy as np

x = [2, 9.1, 12.5]

mean = np.mean(x)

print(np.mean(x) == sum(x) / len(x))

print(mean)

y = [2, 1, 3.0]

y.sort(reverse=True)
print(y)

z = sorted(y, reverse=True)
print(z)

a = [1, "hello", 3.0]

b = tuple(a)
print(b)

c = list(b)
print(c)

first, second, third = b

print(first, "\n", second, "\n", third)

# tuple vs list

china_data = [(2015, 11.06, 1.371), (2014, 10.48, 1.364), (2013, 9.607, 1.357)]

print(china_data)

print(china_data[0][0])

# ZIP AND ENUMERATE

# zip

gdp_data = [9.607, 10.48, 11.06]
years = [2013, 2014, 2015]

z = zip(years, gdp_data)

ls = list(z)

print(ls)

x, y = ls[0]
print(x, y)

print(f"year = {x}, GDP = {y}")


# enumerate

e = enumerate(["a", "b", "c"])
lst = list(e)
print(lst)


# List methods .extend(), .sort(), .append()


# ASSOCIATIVE COLLECTIONS

# dictionaries/dict

china_data = {"country": "China", "year": 2015, "GDP": 11.06, "population": 1.371}

companies = {
    "AAPL": {"bid": 175.96, "ask": 175.98},
    "GE": {"bid": 1047.03, "ask": 1048.40},
    "TVIX": {"bid": 8.38, "ask": 8.40},
}

print(companies)

# getting, setting and updating dict items

print(china_data["year"])

print(f"country = {china_data['country']}, population = {china_data['population']}")

china_data["unemployment"] = "4.05%"

print(china_data)

china_data["unemployment"] = 4.05

print(china_data)

# number of key calue pair in a dict

print(len(china_data))

# get a list of all the keys

ls = list(china_data.keys())

print(ls)

# get a list of all the values

lsv = list(china_data.values())
print(lsv)


# Overwrite and add key-value pairs

more_china_data = {
    "irrigated_land": 690_070,
    "top_religions": {"buddhist": 18.2, "christian": 5.1, "muslim": 1.8},
}


china_data.update(more_china_data)
print(china_data)

# Get the value associated with a key or return a default value
# use this to avoid the NameError

print(china_data.get("irrigated_land", "Data Not Available"))
print(china_data.get("death_rate", "Data Not Available"))

# Sets


s = {1, "hello", 3.0}
print(type(s), s)


print("hello" in s)

s.add(100)
print(s)

s.add("hello")  # does not work as sets have unique items
print(s)

s2 = {"hello", 5, 6, "there"}

print(s.union(s2))  # returns a set with all elements in either s or s2

print(s.intersection(s2))  # returns a set with all elements in both s and s2

print(s.difference(s2))  # returns a set with all elements in s that aren’t in s2

print(
    s.symmetric_difference(s2)
)  # returns a set with all elements in only one of s and s2

# converting lists and tuples to sets and vv

x = [1, 2, 3, 1]
print(set(x))

t = (1, 2, 3, 1)
print(set(t))

print(list(s))
print(tuple(s))


# exercise


lst = enumerate(("foo"))
print(list(lst))


china_data.pop("irrigated_land")
china_data.pop("top_religions")

print(china_data)
