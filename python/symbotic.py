# define list of numbers

import numpy
import pandas

x = [5, 7, 4, 2, 19]
x.append(16)
x = numpy.array(x)

# dictionary key to number
data = { 'input' : x }

df = pandas.DataFrame(x, columns = ['input'])

def print_input(input):
    print(input)

print_input(x)

class printer:
    data
    print_input(self)

