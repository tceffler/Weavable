#include "calculateTotalEnergy.hpp"

void calculateTotalEnergy(LSMSCommunication &comm, LSMSSystemParameters &lsms, LocalTypeInfo &local, CrystalParameters &crystal)
{
  
  //Local now, should be defined as global variables?
  Real totalEnergy = 0.0;
  Real totalPressure = 0.0;

  for (int i=0; i<local.num_local; i++)
  {
    // Local energy and pressure for this atom
    Real energy = 0.0;
    Real pressure = 0.0;

    Real *rhoTotal;
    rhoTotal = new Real[lsms.global.iprpts];


    int isold;

    // Calculate total charge density
    for (int ir=0; ir<local.atom[i].jmt; ir++)
    {
      rhoTotal[ir] = local.atom[i].rhoNew(ir,0);
      if (lsms.n_spin_pola == 2) rhoTotal[ir] += local.atom[i].rhoNew(ir,1);
    }

    for (int is=0; is<lsms.n_spin_pola; is++)
    {
      if (lsms.global.iprint >= 0) printf(" TOTE_FT:: Spin index = %5d*************\n", is);

      // Determine spin direction
      switch (local.atom[i].afm)
      {
        case 1:
          isold = 1 - is;
          break;
        default:
          isold = is;
      }

/*    ===============================================================
      Janake::   Came from KKR-CPA expects: komp, atcon(1->komp).....
                 Set Dummies : Stop run time array bounds warning....
      ===============================================================
*/
      int komp = 1;
      Real atcon = 1.0;
      Real ztotssTemp = local.atom[i].ztotss;     //what for? 

      Real rSphere;
      switch (lsms.mtasa)
      {
        case 1:

          rSphere = local.atom[i].rws;
          break;
        case 2:
          rSphere = local.atom[i].rws;
          break;
        default:
          rSphere = local.atom[i].rInscribed;
      }

      Real rspin = Real(lsms.n_spin_pola);
    
      janake_(&local.atom[i].vr(0,isold), &local.atom[i].vrNew(0,is),
              &rhoTotal[0], &local.atom[i].rhoNew(0,is), &local.atom[i].corden(0,isold),
              &local.atom[i].r_mesh[0], &local.atom[i].rInscribed, &rSphere,
              &local.atom[i].jmt, &local.atom[i].jws, 
              &komp, &atcon,
              &ztotssTemp,
              &local.atom[i].omegaWS,
              &local.atom[i].exchangeCorrelationPotential(0,is), &local.atom[i].exchangeCorrelationEnergy(0,is),
              &local.atom[i].evalsum[is], 
              &local.atom[i].ecorv[isold], &local.atom[i].esemv[isold],
              &energy, &pressure, &rspin,
              &lsms.global.iprpts, 
              &lsms.global.iprint, lsms.global.istop, 32);
    }

    totalEnergy += energy * local.n_per_type[i];
    totalPressure += pressure * local.n_per_type[i];

    delete[] rhoTotal;

  }
/*
  ================================================================
  Perform global sums for the Energy and pressure
  ================================================================
*/
  globalSum(comm, totalEnergy);
  globalSum(comm, totalPressure);

  Real *emad;
  Real *emadp;
  emad = new Real[lsms.n_spin_pola];
  emadp = new Real[lsms.n_spin_pola];

  Real spinFactor;
  switch (lsms.n_spin_pola)
  {
    case 1:
      spinFactor = 1.0;
      break;
    default:
      spinFactor = 0.5;
  }

  for (int is=0; is<lsms.n_spin_pola; is++)
  {
    Real spin = 1.0 - 2.0 * is;

    emad[is] = 0.0;
    emadp[is] = 0.0;

    // calculations of emad and emadp were originally in genpot_c.f in LSMS 1
    if (lsms.mtasa < 1)
    {
      int i = 0;
      emad[is] = spinFactor * (local.atom[i].qInt + spin * local.atom[i].mInt) * local.atom[i].exchangeCorrelationE;
      emadp[is] = -spinFactor * (local.atom[i].qInt + spin * local.atom[i].mInt) * 3.0 * (local.atom[0].exchangeCorrelationE - local.atom[i].exchangeCorrelationV[is]);
      //printf("is, emad, emadp = %5d %35.25f %35.25f\n", is, emad[is], emadp[is]);
    }

    totalEnergy += emad[is];
    totalPressure += emadp[is];
  }

  totalEnergy += lsms.u0;
  totalPressure += lsms.u0;

  printf("Total Energy   = %35.25f\n", totalEnergy);
  printf("Pressure = %35.25f\n", totalPressure);

  lsms.totalEnergy = totalEnergy;

  delete[] emad;
  delete[] emadp;

  return;

}
