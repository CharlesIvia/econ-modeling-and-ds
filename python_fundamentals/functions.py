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

#Returns to scale

#If, for K, L we multiply K,L by a value γ

#If Y2/Y1 < γ => prod fn has decreasing returns to scale
#If Y2/Y1 = γ => prod fn has constant returns to scale
#If Y2/Y1 > γ => prod fn has increasing returns to scale


y1 = cobb_douglas(1, 0.5)
print(y1)

y2 = cobb_douglas(2*1, 2*0.5)
print(y2)

scale = y2 / y1
print(scale)  # => 2, therefore Y2 was exactly double y1

#returns to scale fn


def returns_to_scale(K, L, gamma):
    y1 = cobb_douglas(K, L)
    y2 = cobb_douglas(gamma * K, gamma*L)
    y_ratio = y2 / y1
    return y_ratio / gamma

returns = returns_to_scale(1, 0.5, 2)
print(returns)

#For an example of a production function that is not CRS,
# look at a generalization of the Cobb-Douglas production
# function that has different “output elasticities” for the 2 inputs.


#Multiple returns- marginal product
# how output changes as we change only one of the inputs. We will call this the marginal product.

#MPL defined => F(K,L + e) - F(K,L) / e

#MPL(K,L) = ΔF(K,L) / ΔL
#MPK(K,L) = ΔF(K,L) / ΔK


#Computing MPL and MPK

def marginal_products(K, L, epsilon):
    mpl = (cobb_douglas(K, L + epsilon) - cobb_douglas(K, L)) / epsilon
    mpk = (cobb_douglas(K + epsilon, L) - cobb_douglas(K, L)) / epsilon
    return mpl, mpk

m_products = marginal_products(1, 0.5, 0.0001)
print(m_products)

mpl, mpk = marginal_products(1, 0.5, 0.0001)
print(f"mpl = {mpl}, mpk = {mpk}")

mpl, mpk = marginal_products(2, 0.5, 0.0004)
print(mpl, mpk)

#Using comprehension to get mps of K whilw fixing l
Ks = [1, 2, 3]

mpks = [marginal_products(K, 0.5, 0.0001) for K in Ks]
print(mpks)

#Default and keyword arguments


def cobb_douglas(K, L, alpha=0.33, z=1):
    """
    Computes the production F(K, L) for a Cobb-Douglas production function

    Takes the form F(K, L) = z K^{\alpha} L^{1 - \alpha}
    """
    return z * K**(alpha) * L**(1.0 - alpha)

print(cobb_douglas(1.0, 0.5))
print(cobb_douglas(1.0, 0.5, 0.35, 1.6))
print(cobb_douglas(1.0, 0.5, z=1.5))
