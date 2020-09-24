import numpy as np

x = [2, 9.1, 12.5]

mean = np.mean(x)

print(np.mean(x) == sum(x)/len(x))

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

#tuple vs list

china_data = [(2015, 11.06, 1.371), (2014, 10.48, 1.364), (2013, 9.607, 1.357)]

print(china_data)

print(china_data[0][0])

#ZIP AND ENUMERATE

#zip

gdp_data = [9.607, 10.48, 11.06]
years = [2013, 2014, 2015]

z = zip(years, gdp_data)

ls = list(z)

print(ls)

x, y = ls[0]
print(x, y)

print(f"year = {x}, GDP = {y}")


#enumerate

e = enumerate(["a", "b", "c"])
lst = list(e)
print(lst)