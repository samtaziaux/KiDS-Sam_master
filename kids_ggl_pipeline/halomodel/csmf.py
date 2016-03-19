
#def phi_sub(


def nsm(mass_func, m, M, sigma, alpha, A, M_1, gamma_1, gamma_2,
        b_0, b_1, b_2, Ac2s):
    ns = np.ones(M.size)
    phi_int = phi_s(m, M, alpha, A, M_1, gamma_1, gamma_2,
                    b_0, b_1, b_2, Ac2s)
    for i in xrange(M.size):
        ns[i] = Integrate(phi_int[i], m)
    return ns