import math

def intersection(list1, list2):
    s2 = set(list2)
    return [elem for elem in list1 if elem in s2]

def subtraction(list1, list2):
    s2 = set(list2)
    return [elem for elem in list1 if not (elem in s2)]

def unzip(iterable):
    return zip(*iterable)

def mean(list1d):
    count = len(list1d)
    if count == 0:
        return 0
    float_count = float(count)
    return sum(list1d)/float_count

def add(list1d, values):
    broadcasted = broadcast(list1d, values)
    return map(lambda (x, y): x + y, zip(list1d, broadcasted))

def subtract(list1d, values):
    broadcasted = broadcast(values, list1d)
    return map(lambda (x, y): x - y, zip(list1d, broadcasted))

def sqr(list1d):
    return map(lambda x: x**2, list1d)

def sqrt(list1d):
    return map(lambda x: math.sqrt(x), list1d)

def broadcast(values, list1d):
    list_length = len(list1d)
    values_list = None
    try:
        values_list = list(values)
    except TypeError:
        values_list = [values]
    values_list_len = len(values_list)
    if list_length == values_list_len:
        return values_list
    elif values_list_len <> 1:
        raise ValueError('Cannot broadcast size {0} to {1}'.format(values_list_len, list_length)) 
    return list_length*values_list