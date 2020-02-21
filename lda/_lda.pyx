#cython script
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free


cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)


cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz,
                   double[:] alpha, double[:] eta, double[:] rands, int n_topics):
    # WS - 1 1 1 2 2 (if word 1 appears 3 times in doc 0 and word 2 appears 2 times in doc0)
    # DS - 0 0 0 0 0 (document corresponding to the word)
    # ZS - 0 1 1 1 0 (topic distribution corresponding to the words.)
    # nzw_ - matrix of topic word distribution
    # ndz_ - document topic distribution
    # nz_ - topic distribution
    # alpha - dirichlet prior for distribution over topics. size = no of topics 
    # eta - dirichlet prior for distribution over words. size = no of words (vocab)
    # rands - shuffled random numbers
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0] # total number of words (including the duplicates).
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0] # no of topics
    cdef double eta_sum = 0
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    
    # multithreaded
    # concurrent execution
    with nogil: # global interpreter lock
        # for i in range(eta.shape[0]):
            # eta_sum += eta[i]
        eta_sum = eta.sum()

        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]

            dec(nzw[z, w]) # decrease the counts by 1.
            dec(ndz[d, z])
            dec(nz[z])

            # = lent to p_z = ((cwt[:,word_num] + eta) / (np.sum((cwt), axis = 1) + len(corpus) * eta)) * ((cdt[d,] + alpha) / (sum(cdt[d,]) + K * alpha )) 
            # z = np.sum(p_z) # sum it up to form the denominator for normalization.
            # p_z_ac = np.add.accumulate(p_z/z)   # there is no consideration of the burn in period? 
            # can we use numpy ?
            # okay, this code will be executed faster than python. 
            # https://stackoverflow.com/questions/7799977/numpy-vs-cython-speed
            dist_cum = 0
            for k in range(n_topics):
                # eta is a double so cdivision yields a double
                # dist_cum += (nzw[k, w] + eta[w]) / (nz[k] + eta_sum) * (ndz[d, k] + alpha[k]) # should normalize the second term too?
                # probability of word belonging to a topic X probability of the topic belonging to the document.
                '''this is likely to make the code run slower? sum(ndz[d]) check??'''
                dist_cum += ((nzw[k, w] + eta[w]) / (nz[k] + eta_sum)) * ((ndz[d, k] + alpha[k])/sum(ndz[d]) + n_topics*eta_alpha)
                dist_sum[k] = dist_cum # keep the accumulated sum

            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]

            # randomly picks up a topic.
            # search for r in the accumulated array to generate the new topic.
            z_new = searchsorted(dist_sum, n_topics, r) # binary search

            ZS[i] = z_new # assign the new topic
            inc(nzw[z_new, w]) # increment the counts corresponding to these topics.
            inc(ndz[d, z_new])
            inc(nz[z_new])

        free(dist_sum)


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double eta) nogil:
    cdef int j, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)
        lgamma_alpha = lgamma(alpha)

        ll += n_topics * lgamma(eta * vocab_size)
        for j in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[j])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[j, w] > 0:
                    ll += lgamma(eta + nzw[j, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for j in range(n_topics):
                if ndz[d, j] > 0:
                    ll += lgamma(alpha + ndz[d, j]) - lgamma_alpha
        return ll
