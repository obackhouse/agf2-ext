import numpy as np
import ctypes
from pyscf.agf2 import mpi_helper


def dfragf2_slow_fast_build(qxi, qja, e_occ, e_vir, nmom_max, os_factor=1.0, ss_factor=1.0):
    lib = np.ctypeslib.load_library('agf2_ext/lib/dfragf2_slow_fast.so', 'dfragf2_slow_fast')
    fdrv = lib.AGF2df_build_lowmem_islice

    naux = qxi.shape[0]
    nocc = e_occ.size
    nvir = e_vir.size
    nmo = qxi.size // (naux*nocc)
    assert qxi.size == (naux * nmo * nocc)
    assert qja.size == (naux * nocc * nvir)

    qxi = np.asarray(qxi, order='C')
    qja = np.asarray(qja, order='C')
    e_i = np.asarray(e_occ, order='C') 
    e_a = np.asarray(e_vir, order='C')

    rank, size = mpi_helper.rank, mpi_helper.size

    out = np.zeros(((nmom_max+1)*nmo*nmo))

    start = rank * (nocc * nocc) // size
    end = nocc*nocc if rank == (size-1) else (rank+1) * (nocc*nocc) // size

    fdrv(qxi.ctypes.data_as(ctypes.c_void_p),
         qja.ctypes.data_as(ctypes.c_void_p),
         e_i.ctypes.data_as(ctypes.c_void_p),
         e_a.ctypes.data_as(ctypes.c_void_p),
         ctypes.c_double(os_factor),
         ctypes.c_double(ss_factor),
         ctypes.c_int(nmo),
         ctypes.c_int(nocc),
         ctypes.c_int(nvir),
         ctypes.c_int(naux),
         ctypes.c_int(nmom_max),
         ctypes.c_int(start),
         ctypes.c_int(end),
         out.ctypes.data_as(ctypes.c_void_p))

    out = out.reshape(nmom_max+1, nmo, nmo)

    mpi_helper.barrier()
    mpi_helper.allreduce_safe_inplace(out)

    return out
