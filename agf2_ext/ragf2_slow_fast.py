'''
Fast version of the ragf2_slow.RAGF2 method

WARNING: Numerically unstable for small systems.
'''

import numpy as np
import time
from pyscf.lib import logger
from pyscf import lib, agf2
from agf2_ext import block_lanczos


def build_se_part(gf2, eri, gf_occ, gf_vir, nmom, os_factor=1.0, ss_factor=1.0):
    ''' Builds the auxiliaries.
    '''

    cput0 = (time.clock(), time.time())
    log = logger.Logger(gf2.stdout, gf2.verbose)

    assert type(gf_occ) is agf2.GreensFunction
    assert type(gf_vir) is agf2.GreensFunction

    nmo = gf2.nmo
    nocc = gf_occ.naux
    tol = gf2.weight_tol

    if not (gf2.frozen is None or gf2.frozen == 0):
        mask = agf2.ragf2.get_frozen_mask(agf2)
        nmo -= np.sum(~mask)

    t = np.zeros((2*nmom+2, nmo, nmo))

    ei, ci = gf_occ.energy, gf_occ.coupling
    ea, ca = gf_vir.energy, gf_vir.coupling
    qeri = agf2.ragf2._make_qmo_eris_incore(gf2, eri, (ci,ci,ca))

    eija = lib.direct_sum('i,j,a->ija', ei, ei, -ea)

    p1 = 0
    for i in range(nocc):
        xija = qeri[:,i].reshape(nmo, -1)
        xjia = qeri[:,:,i].reshape(nmo, -1)
        xjia = 2.0 * xija - xjia #TODO: os/ss factors
        eja = eija[i].ravel()

        for n in range(2*nmom+2):
            t[n] = lib.dot(xija, xjia.T, beta=1, c=t[n])
            xjia *= eja[None]

    if not np.all(np.isfinite(t)):
        raise ValueError('Overflow from large moments')

    m, b = block_lanczos.block_lanczos(t, nmom+1, debug=True)
    e, v = block_lanczos.build_from_tridiag(m, b)

    se = agf2.SelfEnergy(e, v, chempot=gf_occ.chempot)
    se.remove_uncoupled(tol=tol)

    if not (gf2.frozen is None or gf2.frozen == 0):
        coupling = np.zeros((nmo, se.naux))
        coupling[mask] = se.coupling
        se = agf2.SelfEnergy(se.energy, coupling, chempot=se.chempot)

    log.timer('se part', *cput0)
    
    return se

class RAGF2(agf2.ragf2_slow.RAGF2):
    build_se_part = build_se_part

    def build_se(self, eri=None, gf=None, os_factor=None, ss_factor=None, se_prev=None):
        if eri is None: eri = self.ao2mo()
        if gf is None: gf = self.gf
        if gf is None: gf = self.init_gf()

        fock = None
        if self.nmom[0] != None:
            fock = self.get_fock(eri=eri, gf=gf)

        if os_factor is None: os_factor = self.os_factor
        if ss_factor is None: ss_factor = self.ss_factor

        facs = dict(os_factor=os_factor, ss_factor=ss_factor)
        gf_occ = gf.get_occupied()
        gf_vir = gf.get_virtual()

        if isinstance(self.nmom[1], (tuple, list)):
            nmom_o, nmom_v = self.nmom[1]
        else:
            nmom_o = nmom_v = self.nmom[1]

        se_occ = self.build_se_part(eri, gf_occ, gf_vir, nmom_o, **facs)
        se_vir = self.build_se_part(eri, gf_vir, gf_occ, nmom_v, **facs)

        se = agf2.aux.combine(se_occ, se_vir)
        se = se.compress(phys=fock, n=(self.nmom[0], None))

        if se_prev is not None and self.damping != 0.0:
            se.coupling *= np.sqrt(1.0-self.damping)
            se_prev.coupling *= np.sqrt(self.damping)
            se = aux.combine(se, se_prev)
            se = se.compress(n=self.nmom)

        return se



if __name__ == '__main__':
    from pyscf import gto, scf, agf2

    mol = gto.M(atom='O 0 0 0; O 0 0 1', basis='6-31g', verbose=4)
    rhf = scf.RHF(mol).run()
    
    gf2_a = agf2.AGF2(rhf, nmom=(2,3)).run()

    gf2_b = RAGF2(rhf, nmom=(2,3)).run()


