"""
o o o o
o x x o
o o x o
o o o o

off limits = 

recursion -> sub problem that simplifies to a base case
"""

def wrapper_function(offlimits):
    # get number of rows and columns
    r,c = size(offlimits)

    # keep track of path
    path = zeros(r, c)

    # take step and recurse
    x_out, y_out, valid = take_step(1, 1, True)

    if path[r][c]:
        return path
    else:
        return "failed"

def take_step(x_in, y_in, valid_in): 
    #  base case
    if x_in == c and y_in == r:
        path[x_in][y_in] = 1
        return x_in, y_in, True

    # can't progress
    if offlimits[x_in][y_in]:
        return x_in, y_in, False

    x_1, y_1, valid1 = take_step(x_in + 1, y_in, valid_in)

    # can move right
    if valid1:
        path[x_1][y_1] = 1
        return x_1, y_1, valid1

    x_2, y_2, valid2 = take_step(x_in, y_in + 1, valid_in)

    # can move down
    if valid2:
        path[x_2][y_2] = 1
        return x_2, y_2, valid2

    
