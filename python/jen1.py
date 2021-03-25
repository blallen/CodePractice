# 1 2 5 12 29 70

"""
1 2 3  4  5  6
1 2 5 12 29 70

"""

def nth_num(N):
    numbers = {}

    numbers[0] = 1
    numbers[1] = 2

    for i in range(2, N):
        numbers[i] = 2 * numbers[i-1] + numbers[i-2]

    return numbers[N-1]

print(nth_num(7))

def recurse(N):
    if N == 0:
        return 0
    
    if N == 1:
        return 1

    return 2 * recurse(N-1) + recurse(N-2)

print(recurse(7))