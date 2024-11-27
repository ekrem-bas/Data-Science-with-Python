
########################################
# Data Analysis with Python
########################################

# - NumPy
# - Pandas
# - Data Visualization: Matplotlib & Seaborn
# - Advanced Functional Exploratory Data Analysis


########################################
# NumPy
########################################

import numpy as np

# concatenating two arrays in python
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(len(a)):
    ab.append(a[i] * b[i])

# concatenating two arrays with NumPy
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

ab = a * b

########################################
# Creating NumPy Arrays
########################################

np_array = np.array([1, 2, 3, 4, 5])
type(np_array)

# Create an array with given number of zeros
np_zeros_array = np.zeros(3, dtype=int)

# Create an array with 10 random integers between 1 and 20
np_random_array = np.random.randint(1, 20, size=10)

# Create a normal distribution array 
# with the given mean, standard deviation and the size 
np_statistical_array = np.random.normal(10, 4, (3,4))

########################################
# Attributes of NumPy Arrays
########################################
# - ndim: number of dimensions
# - shape: shape of array
# - size: total number of elements
# - dtype: type of the data
########################################

np.random.randint(10, size=5)

a.ndim # returns 1 because the array has 1 dimension
a.shape # returns 5, because it is a 1 dimensional array with 5 elements
a.size # returns 5 because it has 5 elements
a.dtype # returns int64 because it contains only integers

########################################
# Reshaping
########################################

np.random.randint(1, 10, size=9)
# reshape it to 2 dimensional array
np.random.randint(1, 10, size=9).reshape((3, 3))

########################################
# Index Selection
########################################

a = np.random.randint(10, size=10)
a[3] # get selected index
a[0: 5] # slicing
a[0] = 999 # change the selected element of the array

m = np.random.randint(0, 10, size=(3, 5))
m[0,2] # get selected index
m[0: 2, 1: 3] # slicing, get 0 to 2nd rows and 1 to 3rd columns
m[2, 3] = 999 # change the selected element of the array

########################################
# Fancy Index
########################################

v = np.arange(0, 30, 3) # create an array with the given intevral, increasing by last argument
v[[1, 2, 3]] # returns the given indexes

########################################
# Conditions in NumPy
########################################

v = np.array([1, 2, 3, 4, 5])
v < 3 # returns T, T, F, F, F
v == 3 # returns F, F, T, F, F

########################################
# Mathematical Operations
########################################

v = np.array([1, 2, 3, 4, 5])
v / 5 # Divide all elements of the NumPy Array
v * 5 / 10 # Multiply all elements of NumPy Array and divide them by 10
v ** 2 # Square all elements of NumPy Array
v + 1 # Add 1 to all elements of NumPy Array

np.subtract(v, 1) # Subtraction
np.add(v, 1) # Addition
np.mean(v) # Mean 
np.sum(v) # Summation
np.min(v) # Min value
np.max(v) # Max value
np.var(v) # Variance

########################################
# Solving equation with two unknowns with NumPy
########################################

# (5 * x0) + x1 = 12
# x0 + (3 * x1) = 10

a = np.array([[5, 1], [1, 3]]) # Coefficients of unknowns
b = np.array([12, 10]) 
np.linalg.solve(a, b) # 1.86, 2.71