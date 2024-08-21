import numpy as np 

# THE BASICS
arr = np.array([[1, 3, 5], [7, 9, 11]])
# Output: [[ 1  3  5]
# 		   [ 7  9 11]]

#ndarray.ndim
print(arr.ndim) # Output: 2

#ndarray.shape
print(arr.shape) # Output: (2, 3)

#ndarray.size
print(arr.size) # Output: 6

#ndarray.dtype
print(arr.dtype) # Output: int32

#ndarray.itemsize
print(arr.itemsize) # Output: 4 (bytes)

#ndarray.data
print(arr.data) # Output: <memory at 0x0000018A4808E380>

#ndarray.astype
print(arr.astype(float)) # Output: [[ 1.  3.  5.]
                         #           [ 7.  9. 11.]]



# ARRAY CREATION
a = np.array([1.1, 3.3, 5.5])
# Output: [1.1 3.3 5.5]
print(a.dtype) # Output: float64

b = np.array([[1, 3, 5], [7, 9, 11]], dtype=complex)
# Output: [[ 1.+0.j  3.+0.j  5.+0.j]
#        [ 7.+0.j  9.+0.j 11.+0.j]]

c=np.zeros((3, 5), dtype='int16')
# Output: [[0 0 0 0 0]
#          [0 0 0 0 0]
#          [0 0 0 0 0]]

zeros_like_arr = np.zeros_like(arr)
print(zeros_like_arr)
# Output: [[0 0 0]
# 		   [0 0 0]]

d=np.ones((3, 5), dtype='float64')
# Output: [[1. 1. 1. 1. 1.]
#         [1. 1. 1. 1. 1.]
#         [1. 1. 1. 1. 1.]]

ones_like_arr = np.ones_like(arr)
print(ones_like_arr)
# Output: [[1 1 1]
#          [1 1 1]]

e=np.empty((2, 2))
# Output: [[5.13828272e-322 8.87134890e-300]
#        [9.95377077e-300 8.87134890e-300]]

empty_like_array = np.empty_like(a)
print(empty_like_array)
# Output: [1.27991129e-152 5.98181694e-154 6.18925211e+223]

f=np.arange(0, 10, 1) #create an array from 0 to 9 with step=1, 10 is exclusive
# Output: [0 1 2 3 4 5 6 7 8 9]

g=np.linspace(0, 1, 5) #create an array with 5 numbers, 1 is inclusive
# Output: [0.   0.25 0.5  0.75 1.  ]


h = np.arange(15).reshape(5,3)
# Output: [[ 0  1  2]
#         [ 3  4  5]
#         [ 6  7  8]
#         [ 9 10 11]
#         [12 13 14]]



# BASIC OPERATIONS
i=np.array([[1, 2, 3], [4, 5, 6]])
j=np.ones((2, 3))
print(i+j) #Output: [[2. 3. 4.]
#				 	 [5. 6. 7.]]

print(i-j) #Output: [[2. 3. 4.]
# 				     [5. 6. 7.]]

print(i**2) #Output: [[ 1  4  9]
#					  [16 25 36]]

print(i<10) #Output: [[ True  True  True]
#                     [ True  True  True]]

#matrix multiplication
A = np.array([[5, 8, -4], [6, 9, -5], [4, 7, -2]])
# Output: [[ 5  8 -4]
#         [ 6  9 -5]
#         [ 4  7 -2]]
B = np.array([2, -3, 1])
# Output: [ 2 -3  1]
C = A.dot(B)
# Output: [-18 -20 -15]

#[[ 1  3  5]
# [ 7  9 11]]
print(arr.sum()) # Output: 36
print(arr.min()) # Output: 1
print(arr.max()) # Output: 11

print(arr.sum(axis=0)) # Output: [ 8 12 16] (summation by each column)
print(arr.sum(axis=1)) # Output: [ 9 27] (summation by each row)



# UNIVERSAL FUNCTIONS
print(np.exp(arr)) 
#Output: [[2.71828183e+00 2.00855369e+01 1.48413159e+02]
#         [1.09663316e+03 8.10308393e+03 5.98741417e+04]]

print(np.cos(arr))
# Output: [[ 0.54030231 -0.9899925   0.28366219]
#         [ 0.75390225 -0.91113026  0.0044257 ]]

print(np.sin(arr))
# Output: [[ 0.84147098  0.14112001 -0.95892427]
# 		  [ 0.6569866   0.41211849 -0.99999021]]

print(np.sqrt(arr))
# Output: [[1.         1.73205081 2.23606798]
# [2.64575131 3.         3.31662479]]

result1 = np.all(arr>0)
print(result1) #Output: True

arr2 = np.zeros((2,2))
result2 = np.all(arr2>0)
print(result2) # Output: False

def func(a):
	return a+2
print(np.apply_along_axis(func,0,arr))
# Output: [[ 3  5  7]
#         [ 9 11 13]]

arr3 = np.array([1, 2, 3, 4, 5])
print(np.argmax(arr3)) # Output: 4
print(np.argmax(arr, axis=0))
# Output: [1 1 1]
print(np.argmax(arr, axis=1))
# Output: [2 2]

print(np.argmin(arr3)) # Output: 0
print(np.argmin(arr, axis=0)) # Output: [0 0 0]
print(np.argmin(arr, axis=1)) # Output: [0 0]

print(np.average(arr)) # Output: 6.0

arr4 = np.array([0, 1, 1, 2, 4, 4, 3, 5])
print(np.bincount(arr4))  #Output: [1 2 1 1 2 1]

print(np.argsort(arr4))
# Output: [0 1 2 3 6 4 5 7]

arr5 = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
print(np.ceil(arr5)) #Output: [2. 3. 4. 5. 6.]

print(np.clip(arr, 5, 9))
# Output: [[5 5 5]
#         [7 9 9]]

x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 2, 3, 4])
print(np.corrcoef(x, y))
# Output: [[1. 1.]
#          [1. 1.]]

print(np.cov(x, y))
# Output: [[2.5 2.5]
#          [2.5 2.5]]

vect1 = np.array([2, -3, 1])
vect2 = np.array([4, -1, 5])
print(np.cross(vect1, vect2))
# Output: [-14  -6  10]

print(np.cumprod(arr))
# Output: [    1     3    15   105   945 10395]

print(np.cumsum(arr))
# Output: [ 1  4  9 16 25 36]

arr6=np.array([1, 3, 4, 6, 7, 9])
print(np.diff(arr6))
# Output: [2 1 2 1 2]

print(np.floor(arr5))
# Output: [1. 2. 3. 4. 5.]

print(np.max(arr))
# Output: 11
print(np.maximum(A, B))
# Output: [[5 8 1]
# [6 9 1]
# [4 7 1]]

print(np.mean(arr)) 
# Output: 6.0
print(np.median(arr))
# Output: 6.0

print(np.min(arr))
# Output: 1
print(np.minimum(A, B))
# Output: [[ 2 -3 -4]
# [ 2 -3 -5]
# [ 2 -3 -2]]

arr7=np.array([0,0,0,1,3,5])
print(np.nonzero(arr7))
# Output: (array([3, 4, 5], dtype=int64),)

print(np.prod(x)) 
# Output: 0

arr8=np.array([1.835, 1.845, 1.856])
print(np.round(arr8, 2))
# Output: [1.84 1.84 1.86]

arr9=np.array([10, 9, 8, 7, 6, 5])
print(np.sort(arr9))
# Output: [ 5  6  7  8  9 10]

print(np.std(arr9))
# Output: 1.707825127659933
print(np.var(arr9))
# Output: 2.9166666666666665

print(np.transpose(arr))
# Output: [[ 1  7]
#          [ 3  9]
#          [ 5 11]]

condition = arr > 5
result = np.where(condition, arr, -1)
print(result)
# Output: [[-1 -1 -1]
#          [ 7  9 11]]



# INDEXING, SLICING AND ITERATING
arr10 = np.arange(1, 16, 2)
print(arr10) # Output: [ 1  3  5  7  9 11 13 15]
print(arr10[2]) # Output: 5
print(arr10[2:5]) # Output: [5 7 9]
print(arr10[::5]) # Output: [ 1 11]
print(arr10[::-1]) # Output: [15 13 11  9  7  5  3  1]

arr11 = np.array([[7, 8, 9],[10, 11, 12], [13, 14, 15]])
# [[ 7  8  9]
# [10 11 12]
# [13 14 15]]

print(arr11[1, 1]) # Output: 11
print(arr11[:, 2]) # Output: [ 9 12 15]
print(arr11[2, :]) # Output: [13 14 15]
print(arr11[1:3, :]) # Output: [[10 11 12]
 							#	[13 14 15]]
print(arr11[-1, :]) # Output: [13 14 15]


arr12 = np.arange(1, 19).reshape((2, 3, 3))
#[[[ 1  2  3]
#  [ 4  5  6]
#  [ 7  8  9]]

# [[10 11 12]
#  [13 14 15]
#  [16 17 18]]]

print(arr12[1, 1, 2]) # Output: 15
print(arr12[0, :, 2]) # Output: [3 6 9]
print(arr12[1, -1, :]) # Output: [16 17 18]
print(arr12[0, ...]) # Output: [[1 2 3]
					 #		    [4 5 6]
                     #          [7 8 9]]

for element in arr12.flat:
	print(element) 
	# Output: 1
	#		  2
	#		  3
	#		  4
	#		  5
	#        ...
	#        18

for index, value in np.ndenumerate(arr12):
	print(index, value)
# Output: (0, 0, 0) 1
#		  (0, 0, 1) 2
#         (0, 0, 2) 3
#         (0, 1, 0) 4
#         (0, 1, 1) 5
#         ... ... ...
#         (1, 2, 1) 17
#         (1, 2, 2) 18



# SHAPE MANIPULATION
print(arr.ravel()) # Output: [ 1  3  5  7  9 11]
print(arr.reshape(3, 2)) # Output: [[ 1  3]
                         #          [ 5  7]
                         #          [ 9 11]]
print(arr.T) # Output: [[ 1  7]
 			 #	        [ 3  9]
             #          [ 5 11]]

arr13 = np.arange(12)
# [ 0  1  2  3  4  5  6  7  8  9 10 11]
#print(np.resize(arr13, (6, 2)))
#[[ 0  1]
# [ 2  3]
# [ 4  5]
# [ 6  7]
# [ 8  9]
# [10 11]]



# STACKING TOGETHER DIFFERENT ARRAYS
arr14 = np.array([1, 3, 5])
arr15 = np.array([7, 9, 11])
print(np.hstack((arr14, arr15))) 
# Output: [ 1  3  5  7  9 11]
print(np.vstack((arr14, arr15))) 
# Output: [[ 1  3  5]
#          [ 7  9 11]]

print(np.column_stack((arr14, arr15)))
# Output: [[ 1  7]
#          [ 3  9]
#          [ 5 11]]

arr16 = np.array([[3, 4, 5], [6, 7, 8]])
#[[3 4 5]
# [6 7 8]]
arr17 = np.array([[9, 10, 11], [12, 13, 14]])
#[[ 9 10 11]
# [12 13 14]]
print(np.concatenate((arr16, arr17), axis=0))
#[[ 3  4  5]
# [ 6  7  8]
# [ 9 10 11]
# [12 13 14]]
print(np.concatenate((arr16, arr17), axis=1))
#[[ 3  4  5  9 10 11]
# [ 6  7  8 12 13 14]]



# SPLITTING ONE ARRAY INTO SEVERAL SMALLER ONES
arr18 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
#[[ 1  2  3  4]
# [ 5  6  7  8]
# [ 9 10 11 12]
# [13 14 15 16]]
print(np.hsplit(arr18, 2))
# Output:
#[array([[ 1,  2],
#        [ 5,  6],
#        [ 9, 10],
#        [13, 14]]), array([[ 3,  4],
#       			        [ 7,  8],
#                           [11, 12],
#                           [15, 16]])]

print(np.vsplit(arr18, 4))
# Output: [array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]]), array([[ 9, 10, 11, 12]]), array([[13, 14, 15, 16]])]

#print(np.array_split(arr18, 3, axis=0))
# Output: [array([[1, 2, 3, 4],
#      			  [5, 6, 7, 8]]), array([[ 9, 10, 11, 12]]), array([[13, 14, 15, 16]])]



# COPIES AND VIEWS
original = np.array([1, 2, 3, 4, 5])
copy = np.copy(original)
copy[-1] = 110

#print(original)  Output: [1 2 3 4 5]
#print(copy)      Output: [  1   2   3   4 110]

view = original.view()
view[-1] = 110
#print(original) Output: [  1   2   3   4 110]
#print(view) 	 Output: [  1   2   3   4 110]




print('\n\n\n\nOriginal array')
print(arr)









