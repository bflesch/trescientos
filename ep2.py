#! /usr/bin/python

from scipy import *
import matplotlib.pyplot as plt
import scipy.ndimage.measurements as meas
import scipy.misc as misc
import argparse

parser = argparse.ArgumentParser(description='ep2 de MAC0300; aplica filtros em imagens JPG grayscale.')
parser.add_argument("image_name", help="imagem a ser usada; se nao informada, usa-se a imagem provida pela funcao lena()", nargs='?', default=False, metavar="imagem")
parser.add_argument("-e", "--explode",action='store_true', required=False, help="ignorar limiares de espectro")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-blur", action='store_true', help="suavizacao por media ponderada")
group.add_argument("-contrast", action='store_true', help="ajuste de contraste por suavizacao de histograma")
group.add_argument("-sharpen", action='store_true', help="realce atraves da aplicacao do operador Lagrangiano")
group.add_argument("-emboss", action='store_true', help="filtro de efeito de alto relevo")
args = parser.parse_args()

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

def histogram(matrix):
   histogram_array = zeros((256),dtype=int32)
   m = ((matrix.shape[0]))
   n = ((matrix.shape[1]))
   for i in range(0, m):
      for j in range(0, n):
         value = matrix[i][j]
         histogram_array[value] = histogram_array[value] + 1
   return histogram_array

def accumulate(histogram_array):
   n = shape(histogram_array)[0]
   current = 0
   for i in range(0, n):
      current = current + histogram_array[i]
      histogram_array[i] = current
   return histogram_array

def first_nonzero(array):
   for i in array:
      if i != 0:
         return i
   return 0

def contrast_normalization(image):
   normalized_image = zeros(image.shape,dtype=int32)
   accumulated_histogram = accumulate(histogram(image))
   minimum_distribution = first_nonzero(accumulated_histogram)
   print "minimum:::", minimum_distribution
   m = image.shape[0]
   n = image.shape[1]
   for i in range(0, m):
      for j in range(0, n):
         normalized_image[i][j] = round(((accumulated_histogram[image[i][j]] - minimum_distribution) * 255.0) / ((m * n) - minimum_distribution) )
   return normalized_image

def blur(image):
   blur_kernel = array([[1.0/16.0,2.0/16.0,1.0/16.0],
                        [2.0/16.0,4.0/16.0,3.0/16.0],
                        [1.0/16.0,2.0/16.0,1.0/16.0]])
   blur = convolute(blur_kernel, image)
   return blur
   
def laplacian(image):
   laplacian_kernel = array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])
   laplacian = convolute(laplacian_kernel, image)
   return laplacian

def sharpen(image):
   return image + laplacian(image)

def emboss(image):
   emboss_kernel = array([[ 4, 0, 0],
                          [ 0, 0, 0],
                          [ 0, 0, -4]])
   emboss = convolute(emboss_kernel, image)
   return emboss

def truncate_option(image):
   if args.explode:
      return image
   return truncate(image)

def truncate(image):
   m = image.shape[0]
   n = image.shape[1]
   for i in range(0, m):
      for j in range(0, n):
         if (image[i][j] > 255):
            image[i][j] = 255
         elif (image[i][j] < 0):
            image[i][j] = 0
   return image

image_name = args.image_name
if image_name:
   image  = misc.imread(args.image_name, True)
   image_path, last_slash, image_name = image_name.rpartition('/') 
else:
   image_name = 'lena.jpg'
   image = lena()
result_base, dot_char, extension = image_name.rpartition('.')

if (args.contrast):
   result = truncate_option(contrast_normalization(image))
   transform = "contrast"
elif (args.blur):
   result = blur(image)
   transform = "blur"
elif (args.sharpen):
   result = truncate_option(sharpen(image))
   transform = "sharpen"
elif (args.emboss):
   result = truncate_option(emboss(image))
   transform = "emboss"
result_name = "".join([result_base, "-", transform, dot_char, extension])
print result_name

#hist_indexes = arange(256)
#plt.gray()
#plt.imshow(result)
#plt.bar(hist_indexes, histo)
#plt.show()
#print(result)
misc.imsave(result_name, result)
