'''
Extensions to the block/band Lanczos algorithms.

WARNING: Numerically unstable for small systems.
'''

import numpy as np
from pyscf import lib


def build_from_tridiag(m, b):
    ''' Build a tridiagonal matrix from a list of on- (M) and off-diagonal (B)
        square matrices.
    '''

    nmo = m[0].shape[0]
    zero = np.zeros_like(m[0], dtype=m[0].dtype)

    h = np.block([[m[i]          if i == j   else
                   b[j]          if j == i-1 else
                   b[i].T.conj() if i == j-1 else zero
                   for j in range(len(m))]
                   for i in range(len(m))])

    e, v = np.linalg.eigh(h[nmo:,nmo:])
    v = np.dot(b[0].T.conj(), v[:nmo])

    return e, v

def force_posdef(x, maxiter=100, tol=0):
    ''' Find the closest positive-definite matrix to x.
        https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite/43244194
    '''

    b = (x + x.T) * 0.5
    _, s, v = np.linalg.svd(b)
    h = np.dot(v.T, np.dot(np.diag(s), v))

    x2 = (b+h) * 0.5
    x3 = (x2 + x2.T) * 0.5

    spacing = np.spacing(np.linalg.norm(x))
    mineig = np.min(np.real(np.linalg.eigvals(x3)))
    i = np.eye(x.shape[0])
    k = 1

    while mineig < tol:
        mineig = np.min(np.real(np.linalg.eigvals(x3)))
        x3 += i * (-mineig * k**2 + spacing)
        k += 1
        if k == maxiter:
            break

    return x3

def matpow(x, n):
    ''' Raise matrix x to the power n.
    '''

    w, v = np.linalg.eigh(x)
    x_out = np.dot(v * w[None]**n, v.T.conj())

    return x_out

def cholesky(x):
    ''' Cholesky decomposition of x.
    '''

    x_out = np.linalg.cholesky(x)

    return x_out

def inv(x):
    ''' Invert x.
    '''
    x_out = np.linalg.inv(x)

    return x_out

class C:
    ''' Class to contain the recursion relations for the vector outer-products.
    '''

    def __init__(self, nmo, force_orth=True, dtype=np.float64):
        self._c = {}
        self.zero = np.zeros((nmo, nmo), dtype=dtype)
        self.eye = np.eye(nmo, dtype=dtype)
        self.force_orth = force_orth
        self.dtype = dtype

    def __getitem__(self, key):
        if key[0] == 0 or key[2] == 0:
            return self.zero
        elif key[0] < key[2]:
            return self._c[key[::-1]].T.conj()
        else:
            return self._c[key]

    def __setitem__(self, key, val):
        if key[0] < key[2]:
            self._c[key[::-1]] = val.T.conj()
        else:
            self._c[key] = val

    def check_sanity(self):
        for (i,n,j), c in self._c.items():
            try:
                if i == j:
                    # Check Hermiticity
                    assert np.allclose(c, c.T.conj())
                if i == 0 or j == 0:
                    # Zeroth iteration Lanczos vector is zero
                    assert np.allclose(c, self.zero)
                elif n == 0 and i == j:
                    # Globally orthogonal Lanczos vectors for i==j
                    assert np.allclose(c, self.eye)
                elif n == 0 and i != j:
                    # Globally orthogonal Lanczos vectors for i!=j
                    assert np.allclose(c, self.zero)
            except AssertionError as e:
                print('\nSanity check failed for C^{%d}_{%d,%d}' % (n,i,j))
                raise e

    def build_11(c, n, env):
        t, binv = env['t'], env['binv']
        c[1,n,1] = np.dot(np.dot(binv.T.conj(), t[n]), binv)
        return c

    def bump_i(c, i, j, n, env):
        # [i+1,n,j] <- [i,1,i] + [i,n,j] + [i,n+1,j] + [i-1,n,j]
        b, binv = env['b'], env['binv']
        if n == 0 and c.force_orth:
            if i+1 != j:
                c[i+1,n,j] = c.zero.copy()
            else:
                c[i+1,n,j] = c.eye.copy()
            return c
        tmp  = c[i,n+1,j].copy()
        tmp -= np.dot(b[i-1], c[i-1,n,j])
        tmp -= np.dot(c[i,1,i], c[i,n,j])
        c[i+1,n,j] = np.dot(binv.T.conj(), tmp)
        return c

    def bump_ij(c, i, n, env):
        # [i+1,n,i+1] <- [i,1,i] + [i,n,i] + [i,n+1,i] + [i,n+2,i] + [i,n,i-1] + [i,n+1,i-1] + [i-1,n,i-1]
        b, binv = env['b'], env['binv']
        if n == 0 and c.force_orth:
            c[i+1,n,i+1] = c.eye.copy()
            return c
        tmp  = c[i,n+2,i].copy()
        tmp -= lib.hermi_sum(np.dot(c[i,n+1,i-1], b[i-1].T.conj()))
        tmp -= lib.hermi_sum(np.dot(c[i,n+1,i], c[i,1,i].T.conj()))
        tmp += lib.hermi_sum(np.dot(np.dot(c[i,1,i], c[i,n,i-1]), b[i-1].T.conj()))
        tmp += np.dot(np.dot(b[i-1], c[i-1,n,i-1]), b[i-1].T.conj())
        tmp += np.dot(np.dot(c[i,1,i], c[i,n,i]), c[i,1,i].T.conj())
        c[i+1,n,i+1] = np.dot(np.dot(binv.T.conj(), tmp), binv)
        return c

def compute_b(i, env, cond_tol=10, maxiter=20, method='eig'):
    c, b, t, nmo = env['c'], env['b'], env['t'], env['nmo']

    if i == 0:
        b2 = t[0]
    else:
        b2  = c[i,2,i].copy()
        b2 -= lib.hermi_sum(np.dot(c[i,1,i-1], b[i-1].T.conj()))
        b2 -= np.dot(c[i,1,i], c[i,1,i].T.conj())
        if i > 1:
            b2 += np.dot(b[i-1], b[i-1].T.conj())

    tol = 10 * np.finfo(c.dtype).eps

    if method == 'eig':
        w, v = np.linalg.eigh(b2)
        w[w < tol] = tol
        bi = np.dot(v * w[None]**0.5, v.T)
        binv = np.dot(v * w[None]**-0.5, v.T)
    elif method == 'chol' or method == 'chol-iter':
        bi = cholesky(force_posdef(b2, tol=tol)).T.conj()
        binv = inv(bi)
    elif method == 'shifted-chol' or method == 'shifted-chol-iter':
        # https://arxiv.org/pdf/1809.11085.pdf
        shift  = np.finfo(dtype).eps 
        shift *= 11 * (nmo*(nmo+(nmo//2)**3) + nmo*(nmo+1))
        shift *= np.trace(b2)
        shift  = np.dot(shift, np.eye(nmo))
        bi = cholesky(force_posdef(b2, tol=tol)).T.conj()
        binv = inv(bi)
    if method.endswith('iter'):
        n = 0
        b2_next = b2
        while np.linalg.cond(b2_next) > cond_tol and n < maxiter:
            b2_next = np.dot(np.dot(binv.T.conj(), b2), binv)
            b_next = cholesky(force_posdef(b2_next, tol=tol)).T.conj()
            bi = np.dot(b_next, bi)
            binv = inv(bi)
            n += 1

    return bi, binv

def block_lanczos(t, nblock, debug=False, b_method='eig'):
    ''' Block Lanczos algorithm using recursion of the moments of the
        occupied or virtual self-energy. nblock = nmom+1, and t must
        index the first 2*nmom+2 moments.
    '''

    nmo = t[0].shape[0]
    m = np.zeros((nblock+1, nmo, nmo), dtype=t[0].dtype)
    b = np.zeros((nblock,   nmo, nmo), dtype=t[0].dtype)
    c = C(nmo, force_orth=False)

    b[0], binv = compute_b(0, locals(), method=b_method)

    for n in range(len(t)):
        c.build_11(n, locals())

    for i in range(1, nblock):
        b[i], binv = compute_b(i, locals(), method=b_method)

        for n in range(2*(nblock-i)-1):
            c.bump_i(i, i, n, locals())

        for n in range(2*(nblock-i)):
            c.bump_ij(i, n, locals())

    if debug:
        c.check_sanity()

    for i in range(1, nblock+1):
        m[i] = c[i,1,i]

    return m, b


if __name__ == '__main__':
    from pyscf import gto, scf, mp, agf2

    nmom = 5

    mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='cc-pvdz', verbose=False)
    rhf = scf.RHF(mol).run()
    mp2 = mp.MP2(rhf).run()
    gf2 = agf2.AGF2(rhf, nmom=(None, None))

    se = gf2.build_se()

    t_occ = se.get_occupied().moment(range(2*nmom+2))
    t_vir = se.get_virtual().moment(range(2*nmom+2))

    se_occ = agf2.SelfEnergy(*build_from_tridiag(*block_lanczos(t_occ, nmom+1)), chempot=se.chempot) 
    se_vir = agf2.SelfEnergy(*build_from_tridiag(*block_lanczos(t_vir, nmom+1)), chempot=se.chempot)
    se = agf2.aux.combine(se_occ, se_vir)

    e_mp2 = agf2.ragf2.energy_mp2(gf2, rhf.mo_energy, se)

    print(mp2.e_corr)
    print(e_mp2)
