#Production fn- Y = F(K,L)

#Cobb-Douglas production fn: Y = zK**αL**1-α

#z = total factor prductivity ( TFP)


#PYTHON FNS

def mean(numbers):
    total = sum(numbers)
    N = len(numbers)
    answer = total / N
    return answer


x = [1, 2, 3, 4]

the_mean = mean(x)
print(the_mean)


#Computing output from a cobb-douglas prod fn

def cobb_douglas(K, L):
    """Create alpha and z"""
    z = 1
    alpha = 0.33
    return z * K**alpha * L**(1-alpha)

output = cobb_douglas(1.0, 0.5)

print(output)