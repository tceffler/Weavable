#ifndef LSMS_SINGLESITESCATTERING_H
#define LSMS_SINGLESITESCATTERING_H

#include "Complex.hpp"
#include "Matrix.hpp"
#include "Array3d.hpp"
#include "AtomData.hpp"
#include "Main/SystemParameters.hpp" 

class SingleScattererSolution {
public:
  Complex energy;
  int kkrsz;

  AtomData *atom;
  Matrix<Complex> tmat_g;
};

class NonRelativisticSingleScattererSolution : public SingleScattererSolution {
public:
  NonRelativisticSingleScattererSolution(){}
  NonRelativisticSingleScattererSolution(LSMSSystemParameters &lsms, AtomData &a, Complex *tmat_g_stor=NULL)
  {
    init(lsms,a,tmat_g_stor);
  }
  void init(LSMSSystemParameters &lsms, AtomData &a, Complex *tmat_g_store=NULL)
  {
    atom=&a;
    kkrsz=a.kkrsz;
    matom.resize(a.lmax+1,2);
    tmat_l.resize(a.kkrsz,a.kkrsz,2);
    zlr.resize(a.r_mesh.size(),a.lmax+1,2);
    jlr.resize(a.r_mesh.size(),a.lmax+1,2);
    if(tmat_g_store!=NULL)
      tmat_g.retarget(a.kkrsz*lsms.n_spin_cant,a.kkrsz*lsms.n_spin_pola,tmat_g_store);
    else
      tmat_g.resize(a.kkrsz*lsms.n_spin_cant,a.kkrsz*lsms.n_spin_pola);
  }
// non relativistic wave functions
  Array3d<Complex> zlr,jlr;
  Matrix<Complex> matom;
  Array3d<Complex> tmat_l;

  Complex ubr[4], ubrd[4];
};

class RelativisticScattererSolution : public SingleScattererSolution {
public:
// relativistic wave functions
  static const int nuzp=2;
  Array3d<Complex> gz,fz,gj,fj;

  std::vector<int> nuz;
  Matrix<int> indz;

  Matrix<Complex> dmat, dmatp;
};


extern "C"
{
  void single_site_tmat_(int *nrel_rel,int *n_spin_cant,int *is,
                         int *n_spin_pola,
                         int * mtasa,Real *rws,
                         int * nrelv,Real *clight,int *lmax,int *kkrsz,
                         Complex *energy,Complex *prel,Complex *pnrel,
                         Real *vr,Real *h,int *jmt,int *jws,Real *r_mesh,
                         Complex *tmat_l,Complex *tmat_g,Complex *matom,
                         Complex *zlr,Complex *jlr,
                         Complex *gz,Complex *fz,Complex *gj,Complex *fj,int *nuz,int *indz,
                         Complex *ubr,Complex *ubrd,Complex *dmat,Complex *dmatp,
                         Real *r_sph,int *iprint,const char *istop);

  void single_scatterer_nonrel_(int *nrelv,double *clight,int *lmax,int *kkrsz,
                                Complex *energy,Complex *prel,Complex *pnrel,
                                double *vr,double *r_mesh,double *h,int *jmt,int *jws,
                                Complex *tmat_l,Complex *matom,
                                Complex *zlr,Complex *jlr,
                                double *r_sph,int *iprpts,int *iprint,char *istop,int istop_len);

}

void calculateSingleScattererSolution(LSMSSystemParameters &lsms, AtomData &atom,
                                      Matrix<Real> &vr,
                                      Complex energy, Complex prel, Complex pnrel,
                                      NonRelativisticSingleScattererSolution &solution);
void calculateScatteringSolutions(LSMSSystemParameters &lsms, std::vector<AtomData> &atom,
                                  Complex energy, Complex prel, Complex pnrel,
                                  std::vector<NonRelativisticSingleScattererSolution> &solution);
#endif
