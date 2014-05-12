import numpy as np
import cvxpy as cp


def H(A, **kwargs):
    return np.transpose(A, **kwargs).conj()


def complex_to_real_matrix(A):

    A_real = np.real(A)
    A_imag = np.imag(A)

    A_ctr = np.vstack([np.hstack([A_real, -A_imag]),
                       np.hstack([A_imag, A_real])])

    return A_ctr


def real_to_complex_vector(b):

    n = b.shape[0]/2
    print b[0:n] + 1j*b[n:]


def echo_beamformer(A_good, A_bad):

    # Convert complex matrices and vectors to real matrices and vectors
    A_good_ctr_H = complex_to_real_matrix(H(A_good))
    A_bad_ctr_H = complex_to_real_matrix(H(A_bad))

    K = A_bad.shape[1]
    h = cp.Variable(2*K)

    # Objective: minimize(norm(h.H * A_good))
    objective = cp.Minimize(sum(cp.square(A_bad_ctr_H * h)))

    # Constraint: sum(h.H * A_good) = 1 + 0*1j
    constraints = [sum((A_good_ctr_H*h)[0:K]) == 1, 
                   sum((A_good_ctr_H*h)[K:]) == 0]

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return real_to_complex_vector(h.value)
