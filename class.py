# import numpy as np
# import scipy
# import matplotlib.pyplot as plt

# Arrays and Lists

# l = [1, 6, 8, 4]
# a = np.array([1, 6, 8, 4])
# Array corresponds roughly 1-1 to arrays in C and C++
# Lists come with bells and whistles, takes more space, but behaves nicely
# like a linked list

# For us, the interface is different

# print(l)
# print(a)

# Lists are concatenated by +, arrays are elementwise summmed with +

# print(l + l)
# print(a + a)

# Accessing elements

# print(a[1])     # Arrays index from 0
# print(a[-1])    # Access last element
# print(a[-2])    # Access second from last element

# print(a[1:])    # Everything after and including the 1st
# print(a[: -2])  # Everything before the -2th
                  # [included:excluded)

# Array generation

# z = np.zeros((5,4,2)) # 5x4x2 Tensor with zeros
# print(z[1,2,1])
# print(z)

# print(np.zeros((2,4)))
# print(np.ones((2,4)))
# print(np.eye(4))


# print(scipy.sparse.diags(diagonals=[-1,2,-1],
                         # offsets=[-1,0,1],
                         # shape=(4,4)).todense())

# Matrix vector multiplication

# b = np.ones((4,4))
# a = np.eye(4)
# print(a * b) # Elementwise multiplication
# print(a @ b) # Matrix multiplication

# Pretty print

# i = 5
# print('i + 2 = {i+2}')
# versus
# print(f'i + 2 = {i+2}')
# print(f'pi = {np.pi}')
# print(f'pi = {np.pi: .4f} (four digits)') # format string

# print(list(range(10)))

# do not mix tabs and spaces

# dictionaries

# d = {'Steve': 40, 'Bill': 32}
# print(d)

# print(d['Bill'])

# for k in d:
#     print(k)

# for k in d.items():
#     print(k)

# print('---')
# for i, v in enumerate(np.array([1, 7, 2, 1])):
#     print(i, v)

# print(enumerate(np.array([1, 7, 2, 1])))

# List comprehensions

# print(    [ i for i in range(5) ]    )
# print(    [ i**2+4 for i in range(5) ]    )

# l = list(range(5))

# print(  [ np.sin(i) for i in l] )

# def g(x: np.float64, y: np.float64) -> np.float64: # python ignores this but good to document
    # z = x + 2 + y
    # return z

# print( g(2.,3.) )

# x = np.array([1., 2, 3, 4])
# print (x.dtype)
# print(x)
# what if we want to define a function that acts on the elements?
# print(np.sin(x)) # componentwise sine

# loops are slow in python, try to avoid them, python checks the type during each iteration of the loop

# Plotting


# x = np.linspace(-5, 5, 100)
# y = np.sin(x)
# plt.plot(x, y, 'r-') # same as matlab for color formatting
# plt.show()


# classes

# a = np.array([1,2])
# print(a.size)   # property
# print(a.shape)  # property
# print(a.sum())  # function from this class
