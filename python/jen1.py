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

def recurse_all_cases(N, x1, x2, c1, c2):
    if N == 1:
        return x1

    if N == 2:
        return x2

    x_Nm1 = recurse_all_cases(N-1, x1, x2, c1, c2)
    x_Nm2 = recurse_all_cases(N-2, x1, x2, c1, c2)

    answer = c1 * x_Nm1 + c2 * x_Nm2

    return answer
