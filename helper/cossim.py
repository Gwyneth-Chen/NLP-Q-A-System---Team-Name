def cosine_similarity_unigram(a,b):
    """
        takes in 2 tuples/lists of strings
    """
    vocab = list(set(a+b))

    a = {token: a.count(token) for token in vocab}
    b = {token: b.count(token) for token in vocab}

    def dot(a,b):
        out = 0
        for k in a:
            out += a[k]*b[k]
        return out

    def magnitude(v):
        return sum([val**2 for val in v.values()]) ** 0.5

    return dot(a,b)/magnitude(a)/magnitude(b)

def cosine_similarity_bigram(a,b):
    """
        use of bigrams instead of unigrams to compare
    """

    def getvec(v):
        d = {}
        for i in range(len(v)-1):
            bi = v[i], v[i+1]

            if bi not in d:
                d[bi] = 0
            
            d[bi] += 1
        return d

    a,b = getvec(a), getvec(b)

    def combine(a,b):
        """
            make sure both vectors have same keys
        """
        for vec in [a,b]:
            for token in list(a.keys()) + list(b.keys()):
                if token not in vec:
                    vec[token] = 0

    combine(a,b)

    def dot(a,b):
        out = 0
        for k in a:
            out += a[k]*b[k]
        return out

    def magnitude(v):
        return sum([val**2 for val in v.values()]) ** 0.5

    ma = magnitude(a)
    mb = magnitude(b)

    if ma==0 or mb==0:
        return 0

    return dot(a,b) / (ma*mb)  

