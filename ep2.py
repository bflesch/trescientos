#! /usr/bin/python

from scipy import *
import matplotlib.pyplot as plt

def convolution_step(convolution, matrix, i, j):
   element = 0.0
   for l in range(-1, 2):
      for m in range(-1,2):
         element = element + ((convolution[l+1][m+1]) * (matrix[i+l][j+m]))
   return element

def convolute(convolution, matrix):
   matrix_copy = zeros(matrix.shape,dtype=int32)
   m = ((matrix.shape[0])-1)
   n = ((matrix.shape[1])-1)
   for i in range(1, m):
      for j in range(1, n):
         matrix_copy[i][j] = convolution_step(convolution, matrix, i, j)
   return matrix_copy

image = lena()

blur_kernel = array([[1.0/16.0,2.0/16.0,1.0/16.0],
                     [2.0/16.0,4.0/16.0,3.0/16.0],
                     [1.0/16.0,2.0/16.0,1.0/16.0]])
blur = convolute(blur_kernel, image)
laplacian_kernel = array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
laplacian = convolute(laplacian_kernel, image)

plt.gray()
plt.imshow(image + laplacian)
plt.show()
