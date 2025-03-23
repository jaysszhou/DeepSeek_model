import torch
import numpy
import math
def print_hi(name):
    print(f'Hi,{name}')

def softmax(inMatrix):
    m,n = numpy.shape(inMatrix)
    outMatrix = numpy.asmatrix(numpy.zeros((m,n)))
    soft_sum = 0
    for idx in range(0,n):
        outMatrix[0,idx] = math.exp(inMatrix[0,idx])
        soft_sum += outMatrix[0,idx]
    for idx in range(0,n):
        outMatrix[0,idx] = outMatrix[0,idx] / soft_sum
    return outMatrix

if __name__ == '__main__':
    print_hi('hello world')
    a = numpy.array([[1,2,1,2,1,1,3]])
    b = softmax(a)
    print(b)
    
    result = torch.tensor(1) + torch.tensor(2.0)
    print(result)
    