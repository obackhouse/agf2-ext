#include<stdlib.h>
#include<assert.h>
#include<math.h>

#include "vhf/fblas.h"
#include "ragf2.h"


void AGF2df_build_islice(double *qxi,
                         double *qja,
                         double *e_i,
                         double *e_a,
                         double os_factor,
                         double ss_factor,
                         int nmo,
                         int nocc,
                         int nvir,
                         int naux,
                         int nmom_max,
                         int istart,
                         int iend,
                         double *out)
{
    const double D0 = 0.0;
    const double D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';

    const int nja = nocc * nvir;
    const int nxi = nmo * nocc;
    const double fpos = os_factor + ss_factor;
    const double fneg = -1.0 * ss_factor;

#pragma omp parallel
{
    double *qa = calloc(naux*nvir, sizeof(double));
    double *qx = calloc(naux*nmo, sizeof(double));
    double *eja = calloc(nocc*nvir, sizeof(double));
    double *xia = calloc(nmo*nocc*nvir, sizeof(double));
    double *xja = calloc(nmo*nocc*nvir, sizeof(double));

    double *out_priv = calloc((nmom_max+1)*nmo*nmo, sizeof(double));

    int i, n;

#pragma omp for
    for (i = istart; i < iend; i++) {
        // build qx
        AGF2slice_01i(qxi, naux, nmo, nocc, i, qx);

        // build qa
        AGF2slice_0i2(qja, naux, nocc, nvir, i, qa);

        // build xija = xq * qja
        dgemm_(&TRANS_N, &TRANS_T, &nja, &nmo, &naux, &D1, qja, &nja, qx, &nmo, &D0, xja, &nja);

        // build xjia = xiq * qa
        dgemm_(&TRANS_N, &TRANS_T, &nvir, &nxi, &naux, &D1, qa, &nvir, qxi, &nxi, &D0, xia, &nvir);
        //printf("%13.9f %13.9f\n", xja[10], xia[10]); fflush(stdout);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], e_i, e_a, nocc, nvir, eja);

        // inplace xjia = 2 * xija - xjia
        AGF2sum_inplace(xja, xia, nmo*nja, fpos, fneg);

        // out_0xy += xija * (2 yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, out_priv, &nmo);

        for (n = 1; n <= nmom_max; n++) {
            // inplace xija = eija * xija
            AGF2prod_inplace_ener(eja, xja, nmo, nja);

            // out_nxy += xija * eija * (2 yija - yjia)
            dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nja, &D1, xia, &nja, xja, &nja, &D1, &(out_priv[n*nmo*nmo]), &nmo);
        }
    }

    free(qa);
    free(qx);
    free(eja);
    free(xia);
    free(xja);

#pragma omp critical
    for (i = 0; i < ((nmom_max+1)*nmo*nmo); i++) {
        out[i] += out_priv[i];
    }

    free(out_priv);
}
}


void AGF2df_build_lowmem_islice(double *qxi,
                                double *qja,
                                double *e_i,
                                double *e_a,
                                double os_factor,
                                double ss_factor,
                                int nmo,
                                int nocc,
                                int nvir,
                                int naux,
                                int nmom_max,
                                int start,
                                int end,
                                double *out)
{
    const double D0 = 0.0;
    const double D1 = 1.0;
    const char TRANS_T = 'T';
    const char TRANS_N = 'N';
    const int one = 1;

    const double fpos = os_factor + ss_factor;
    const double fneg = -1.0 * ss_factor;

#pragma omp parallel
{
    double *qx_i = calloc(naux*nmo, sizeof(double));
    double *qx_j = calloc(naux*nmo, sizeof(double));
    double *qa_i = calloc(naux*nvir, sizeof(double));
    double *qa_j = calloc(naux*nvir, sizeof(double));
    double *xa_i = calloc(nmo*nvir, sizeof(double));
    double *xa_j = calloc(nmo*nvir, sizeof(double));
    double *ea = calloc(nvir, sizeof(double));

    double *out_priv = calloc((nmom_max+1)*nmo*nmo, sizeof(double));

    int i, j, ij, n;

#pragma omp for
    for (ij = start; ij < end; ij++) {
        i = ij / nocc;
        j = ij % nocc;

        // build qx_i
        AGF2slice_01i(qxi, naux, nmo, nocc, i, qx_i);

        // build qx_j
        AGF2slice_01i(qxi, naux, nmo, nocc, j, qx_j);

        // build qa_i
        AGF2slice_0i2(qja, naux, nocc, nvir, i, qa_i);

        // build qa_j
        AGF2slice_0i2(qja, naux, nocc, nvir, j, qa_j);

        // build xija
        dgemm_(&TRANS_N, &TRANS_T, &nvir, &nmo, &naux, &D1, qa_i, &nvir, qx_j, &nmo, &D0, xa_i, &nvir);

        // build xjia
        dgemm_(&TRANS_N, &TRANS_T, &nvir, &nmo, &naux, &D1, qa_j, &nvir, qx_i, &nmo, &D0, xa_j, &nvir);

        // build eija = ei + ej - ea
        AGF2sum_inplace_ener(e_i[i], &(e_i[j]), e_a, one, nvir, ea);

        // inplace xjia = 2 * xija - xjia
        AGF2sum_inplace(xa_j, xa_i, nmo*nvir, fpos, fneg);

        // out_0xy += xija * (2 * yija - yjia)
        dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nvir, &D1, xa_j, &nvir, xa_i, &nvir, &D1, out_priv, &nmo);

        for (n = 1; n <= nmom_max; n++) {
            // inplace xija = eija * xija
            AGF2prod_inplace_ener(ea, xa_i, nmo, nvir);

            // out_nxy += xija * (2 * yija - yjia)
            dgemm_(&TRANS_T, &TRANS_N, &nmo, &nmo, &nvir, &D1, xa_j, &nvir, xa_i, &nvir, &D1, &(out_priv[n*nmo*nmo]), &nmo);
        }
    }

    free(qx_i);
    free(qx_j);
    free(qa_i);
    free(qa_j);
    free(xa_i);
    free(xa_j);
    free(ea);

#pragma omp critical
    for (i = 0; i < ((nmom_max+1)*nmo*nmo); i++) {
        out[i] += out_priv[i];
    }

    free(out_priv);
}
}
