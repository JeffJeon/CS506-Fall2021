def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x:list, y:list):
    return sum(abs(val1-val2) for val1, val2 in zip(x,y))

def jaccard_dist(x:list, y:list):
    intersection = len(list(set(x).intersection(y)))
    union = (len(x) + len(y)) - intersection
    return float(intersection) / union

def cosine_sim(x, y):

    xx = sum([i**2 for i in x ]) ** (0.5)
    yy = sum([i**2 for i in y ]) ** (0.5)
    numerator  = sum([x[i]*y[i] for i in range(len(x))])
    denominator = xx * yy

    return numerator / denominator

# Feel free to add more

