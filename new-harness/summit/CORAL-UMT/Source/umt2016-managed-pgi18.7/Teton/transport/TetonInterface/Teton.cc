#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include "DBC.hh"
#include <stdexcept>
#include <iostream>
#include "nvToolsExt.h"
#include "transport/TetonInterface/Teton.hh"
#include "transport/TetonInterface/TetonNT.hh"
//#include "transport/EIPhysics/Rad3T/Rad3TCommon.hh"
#include "scstd.h"
using std::vector;
using std::cout;
using std::endl;

// using namespace KullEICoupling;

#undef max

extern "C" void Timer_Beg(const char *);
extern "C" void Timer_End(const char *);
extern "C" void Timer_Print(void);


#include <new> // bad_alloc, bad_array_new_length

template <class T> struct Mallocator2 {

  typedef T value_type;
  Mallocator2() { }; // default ctor not required
  template <class U> Mallocator2(const Mallocator2<U>&);// noexcept { };
  template <class U> bool operator==(
				     const Mallocator2<U>&) const { return true; }
  template <class U> bool operator!=(
				     const Mallocator2<U>&) const { return false; }

  T * allocate(const size_t n) const {
    if (n == 0) { return NULL; }
    if (n > static_cast<size_t>(-1) / sizeof(T)) {
      printf ("Error in Mallocator2\n");
    }
    printf ("Mallocator2 DIS!\n");
    void * const pv = malloc(n * sizeof(T));
    if (!pv) { throw std::bad_alloc(); }
    return static_cast<T *>(pv);
  }

  void deallocate(T * const p, size_t) const {
    free(p);
  }

};


extern "C"
{

  //#define cudaFuncCachePreferShared 1
  //#define cudaFuncCachePreferL1 2
  //extern int cudaDeviceSetCacheConfig2(int);
  //extern int cudaDeviceSetCacheConfig(int);

#if !defined LINUX && !defined BGP
    extern void *
    bsearch (const void *,
             const void *, 
             unsigned long,
             unsigned long,
             int(*) (const void *, const void *));

    extern void 
    qsort (void *, 
           unsigned long, 
           unsigned long,
           int(*) (const void *,const void *));
   
#endif
    extern int 
    MPI_Barrier (MPI_Comm);

#include "TetonVarDecls.hh"
    void 
    F77_ID(getedits_, getedits, GETEDITS)
        (int *, int *, int *, int *, int *,
         int *, int *, double *,
         double *, double *, double *, double *,
         double *, double *, double *, double *);
    void
    F77_ID(settimestep_, settimestep, SETTIMESTEP)
        (double *, double *, double *, double *);

    void
    F77_ID(setenergyedits_, setenergyedits, SETENERGYEDITS)
        (double *, double *, double *, double *, 
         double *, double *, double *, double *);

    void
    F77_ID(setsnorder_, setsnorder, SETSNORDER)
        (int *, int *, double *);

    void
    F77_ID(constructboundary_, constructboundary, CONSTRUCTBOUNDARY)
        (int *, int *, int *, int *);

    void 
    F77_ID(constructitercontrols_, constructitercontrols, CONSTRUCTITERCONTROLS) 
        (int *, int *, int *,
         double *, double *, double *);

    void
    F77_ID(constructdtcontrols_, constructdtcontrols, CONSTRUCTDTCONTROLS)
        (double *, double *, double *, double *, double *);

    void
    F77_ID(constructgeometry_, constructgeometry, CONSTRUCTGEOMETRY)
        ();

    void
    F77_ID(constructmaterial_, constructmaterial, CONSTRUCTMATERIAL)
        ();

    void
    F77_ID(constructprofile_, constructprofile, CONSTRUCTPROFILE)
        (int *);

    void
    F77_ID(constructquadrature_, constructquadrature, CONSTRUCTQUADRATURE)
        (int *, int *);

    void
    F77_ID(setmaterialmodule_, setmaterialmodule, SETMATERIALMODULE)
        (double *, double *, double *, double *, double *,
         double *, double *, double *, double *); 

    void
    F77_ID(seteditormodule_, seteditormodule, SETEDITORMODULE)
        (double *, double *, double *, double *);

    void
    F77_ID(constructeditor_, constructeditor, CONSTRUCTEDITOR)
        ();

    void
    F77_ID(setgeometry_, setgeometry, SETGEOMETRY)
        (int *, int *, int *, 
         int *, int *, int *, double *); 

    void
    F77_ID(setzone_, setzone, SETZONE)
        (int *, int *, int *, int *, int *, int *);

    void
    F77_ID(getgeometry_, getgeometry, GETGEOMETRY)
        ();

    void
    F77_ID(rtinit_, rtinit, RTINIT)
        (double *, double *);

    void 
    F77_ID(radtr_, radtr, RADTR)
        ( double *,   // PSIR
          double *,   // PHIR
          double *,   // RadEnergyDensity 
          double *);  // angleLoopTime

    void
    F77_ID(pinmem_, pinmem, PINMEM)
      ( double *, // psir
        double *); // Phi


    void 
    F77_ID(radtr_, radtr, RADTR)
        ( double *,   // PSIR
          double *,   // PHIR
          double *,   // RadEnergyDensity 
          double *);  // angleLoopTime

}

template <typename Mesh>
Teton<Mesh>::Teton():
// Most of these values are crucial for dimensioning,
// make sure that they have reasonable values
// These need to be in the same order as declared in the header, or the compiler will complain -AB
        ndim(3), my_node(0), ncycle(0),
        nbelem(0), nzones(0), ncornr(0),
        npnts(0), nfaces(0), nbedit(0),
        ngr(0), noutmx(20), ninmx(1), ngdamx(7),
        nprof(0), GTAorder(2), npsi(0), 
        EnergyRadiation(0.0), EnergyMaterial(0.0),
        EnergyIncident(0.0), EnergyEscaped(0.0), EnergyExtSources(0.0),
        EnergyCheck(0.0), deltaEsrc(0.0), deltaEhyd(0.0),
        tfloor(2.5e-5), tmin(5.0e-03), dtrad(1.0e-03),
        epstmp(1.0e-4), epsinr(1.0e-4), epsgda(1.0e-5), 
        delte(0.4), deltr(0.4), timerad(0.0), 
        dtrmn(1.0e-04), dtrmx(0.1),  ncomm(0), 
        nbshare(0), mPartListPtr(0),angleLoopTime(0.0)
{
// CharStar8 are structs, not classes!!!  Just do the
// data transfer (blank padded)
    strncpy (ittyp.data,   "timedep ", 8);
    strncpy (iaccel.data,  "gda     ", 8);
    strncpy (iscat.data,   "on      ", 8);
    strncpy (itimsrc.data, "exact   ", 8);
    strncpy (decomp_s.data,"off     ", 8);
}

// ------------------------------------------------------------
// resize
//
// Make sure all arrays have a reasonable dimension.  Some
// derived scalars are computed here as well.
// ------------------------------------------------------------

template <typename Mesh>
void
Teton<Mesh>::resize() {

    int isSphere = 0, isCylinder = 0, isSlab =0, isXy = 0, isRz = 0;
    int isXyz = 0;
    int D_ncornr, D_nzones;
    long int D_ncornr_npsi;
    int D_nbedit, D_nbedit_ngr;
    long int D_ncornr_ngr;
    int D_nbedit1_ngr, D_npnts_ndim;
    int D_nzones_ngr, D_zones_ndim, D_ncornr_ndim, D_ngr1;
// Set  ndim
    isSphere   = ( strncmp(igeom.data, "sphere  ", 8) == 0 );
    isCylinder = ( strncmp(igeom.data, "cylinder", 8) == 0 );
    isSlab     = ( strncmp(igeom.data, "slab    ", 8) == 0 );
    isXy       = ( strncmp(igeom.data, "xy      ", 8) == 0 );
    isRz       = ( strncmp(igeom.data, "rz      ", 8) == 0 );
    isXyz      = ( strncmp(igeom.data, "xyz     ", 8) == 0 );
   
    if ( isSphere || isCylinder || isSlab ) {
        ndim  = 1;
    } 
    else if ( isXy || isRz ) {
        ndim  = 2;
        maxcf = 2;
    } 
    else if ( isXyz ) {
        ndim  = 3;
        maxcf = 3;
    } 
    else {
        ndim  = 0;
    }


    D_ncornr            = std::max(ncornr, 0);
    //cout<<"ncornr = "<<ncornr<<endl;
    D_nzones            = std::max(nzones, 0);
    //cout<<"ncornr ="<<ncornr<<"npsi ="<<npsi<<endl;
    long temp = (long)ncornr*(long)npsi;
    //cout<<"temp = "<<temp<<endl;
    D_ncornr_npsi       = temp; //std::max(temp, 0); //ncornr*npsi overflows int
    //cout<<"D_ncornr_npsi ="<<D_ncornr_npsi<<endl;

    temp = (long)ncornr*(long)ngr;
    //cout<<"temp = "<<temp<<endl;
    
    D_ncornr_ngr        = temp;//std::max(temp, 0);
    D_ncornr_ndim       = std::max(ncornr * ndim, 0);
    D_nbedit            = std::max(nbedit, 0);
    D_nbedit_ngr        = std::max(nbedit * ngr, 0);
    D_nbedit1_ngr       = std::max((nbedit+1) * ngr, 0);
    D_npnts_ndim        = std::max(npnts * ndim, 0);
    D_nzones_ngr        = std::max(nzones * ngr, 0);
    D_zones_ndim        = std::max(nzones * ndim, 0);
    D_ngr1              = std::max(ngr+1, 0);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    //printf("[%d] Starting allocating pinned host memory...%ld\n", myrank, (D_ncornr_npsi+D_ncornr_ngr)*8 );
    psir.resize(D_ncornr_npsi);
    Phi.resize(D_ncornr_ngr);
    //printf("[%d] Done with allocating pinned host memory...\n", myrank);
    RadEnergyDensity.resize(D_nzones_ngr);
    RadiationForce.resize(D_ncornr_ndim);
    RadiationFlux.resize(D_zones_ndim);
    tec.resize(D_ncornr);
    RadEnergyEscRate.resize(D_nbedit1_ngr);
    RadEnergyIncRate.resize(D_nbedit1_ngr);
    RadEnergyEscape.resize(D_nbedit1_ngr);
    RadEnergyIncident.resize(D_nbedit1_ngr);
    RE_Escape.resize(D_nbedit);
    RE_Incident.resize(D_nbedit);
    RE_EscapeRate.resize(D_nbedit);
    RE_IncidentRate.resize(D_nbedit);
    RE_EscapeSpectrum.resize(D_nbedit_ngr);
    RE_IncidentSpectrum.resize(D_nbedit_ngr);
    RE_EscapeRateSpectrum.resize(D_nbedit_ngr);
    RE_IncidentRateSpectrum.resize(D_nbedit_ngr);
    denez.resize(D_nzones);
    trz.resize(D_nzones);
    tez.resize(D_nzones);
    px.resize(D_npnts_ndim);
    siga.resize(D_nzones_ngr);
    sigs.resize(D_nzones_ngr);
    cve.resize(D_nzones);
    rho.resize(D_nzones);
    SMatEff.resize(D_nzones);
    gnu.resize(D_ngr1);

printf("[%d] Done with resize!\n", myrank);
}

// ------------------------------------------------------------
//   CXX_RADTR  - Control program for radiation transport. 
// ------------------------------------------------------------

template <typename Mesh>
double
Teton<Mesh>::cxxRadtr() {

// ------------------------------------------------------------
// Move time ahead
// ------------------------------------------------------------
    ncycle++;

    F77_ID(settimestep_, settimestep, SETTIMESTEP)
        (&dtrad, &timerad, &tfloor, &tmin);

    F77_ID(setenergyedits_, setenergyedits, SETENERGYEDITS)
        (&EnergyRadiation, &EnergyMaterial, &EnergyIncident, 
         &EnergyEscaped, &EnergyExtSources, 
         &EnergyCheck, &deltaEsrc, &deltaEhyd);

// ------------------------------------------------------------
// Run the step
// ------------------------------------------------------------
    F77_ID(radtr_, radtr, RADTR)
        ( &psir[0],               // double *
          &Phi[0],                // double *
          &RadEnergyDensity[0],   // double *
          &angleLoopTime );       // double *

    timerad += dtrad;

    F77_ID(getedits_, getedits, GETEDITS)
        (&noutrt, &ninrt, &ngdart,
         &TrMaxZone, &TeMaxZone, &TrMaxNode, &TeMaxNode, 
         &dtrad, &TrMax, &TeMax, &EnergyRadiation, &EnergyMaterial,
         &EnergyIncident, &EnergyEscaped, &EnergyExtSources, &EnergyCheck);

    
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >=1100 )
    long int stringSize=8;
    F77_ID(getrunstats_, getrunstats, GETRUNSTATS)
        (&ConvControlNode, &ConvControlZone,
         &DtControlNode, &DtControlZoneTr4,
         &DtControlZoneTe,
         &ConvControlError, &ConvControlTr,
         &ConvControlTe, &ConvControlRho,
         &ConvControlCve, &ConvControlEdep,
         &DtControlChangeTr4, &DtControlChangeTe, 
         &DtControlTr, &DtControlTe,
         &DtControlTrOld, &DtControlTeOld,
         &CommTimeCycle, &CommTimeTotal,
         ConvControlReason.data, DtControlReason.data,
         &stringSize,&stringSize);
#else
    F77_ID(getrunstats_, getrunstats, GETRUNSTATS)
        (&ConvControlNode, &ConvControlZone,
         &DtControlNode, &DtControlZoneTr4,
         &DtControlZoneTe,
         &ConvControlError, &ConvControlTr,
         &ConvControlTe, &ConvControlRho,
         &ConvControlCve, &ConvControlEdep,
         &DtControlChangeTr4, &DtControlChangeTe, 
         &DtControlTr, &DtControlTe,
         &DtControlTrOld, &DtControlTeOld,
         &CommTimeCycle, &CommTimeTotal,
         ConvControlReason.data, DtControlReason.data);
#endif

    return dtrad;
}

// ------------------------------------------------------------
// setBCs
//
// Associate boundary conditions with Teton (call only once!)
// ------------------------------------------------------------
template <typename Mesh>
void
Teton<Mesh>::setBCs(std::vector<TetonBoundary<Mesh> > &BCList) {

    int position = 1;
    nbc = BCList.size();

// ------------------------------------------------------------
//  Loop over boundary conditions and build the profile
// list
// ------------------------------------------------------------
    for(int i=0; i<nbc; ++i) {
        setAProfile (BCList[i].profile, position);
    }

}

// ------------------------------------------------------------
// setVSs
//
// Associate volume sources with Teton (call only once!)
// ------------------------------------------------------------
template <typename Mesh>
void
Teton<Mesh>::setVSs(std::vector<TetonVolumeSource> &VSList) {

    int position = 0;
    nvs = VSList.size();

// ------------------------------------------------------------
// Loop over volume sources and build the profile list
// ------------------------------------------------------------
    for(int i=0; i<nvs; ++i) {
        setAProfile (VSList[i].profile, position);
    }

}


// ------------------------------------------------------------
// setAProfile
//
// Unpack a profile's time and value data.  This routine
// builds the Profile Module.
// ------------------------------------------------------------
template <typename Mesh>
void
Teton<Mesh>::setAProfile(TetonProfile &P, int position) {

// If empty, no work
    int noshape = 0, sourceFds = 0, groups, NumTimes, NumValues;
    int i=0, j=0, g=0, NumInterpValues;
    double Multiplier;
    CharStar8 Type, Shape, Location;
    std::vector<double> Times, Values;

    noshape = P.shapeless();
    if ( noshape  ) {
        return;
    }

    P.profID = ++nprof;

    NumTimes = P.tim(ngr);

    sourceFds = P.typeIs("fds");
    if ( sourceFds ) {
        groups = ngr;
    } 
    else {
        groups = 1;
    }

    if ( position ) {
        strncpy(Location.data,"boundary",8);
    }
    else {
        strncpy(Location.data,"region  ",8);
    }
    long int stringLen=8;
    
    NumValues       = NumTimes*groups;
    NumInterpValues = ngr;

    Times.resize(NumTimes);
    Values.resize(NumValues);

    for ( i=0; i<NumTimes; ++i) {
        Times[i] = P.timetemps[j++];
        for( g=0; g<groups; ++g ) {
            Values[i*groups+g] = P.timetemps[j++];
        }
    }

    Multiplier = P.prmult;
    Type       = P.type;
    Shape      = P.pshape;

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >=1100 )
    F77_ID(addprofile_, addprofile, ADDPROFILE)
        (&nprof, &NumTimes, &NumValues, 
         &NumInterpValues, &Multiplier,
         Location.data, Type.data, Shape.data,
         &Times[0], &Values[0],
         &stringLen,&stringLen,&stringLen);
#else
    F77_ID(addprofile_, addprofile, ADDPROFILE)
        (&nprof, &NumTimes, &NumValues, 
         &NumInterpValues, &Multiplier,
         Location.data, Type.data, Shape.data,
         &Times[0], &Values[0]);
#endif

    return;
}


// ------------------------------------------------------------
// setGroups
//
// Associate frequency group info with Teton (call once)
// ------------------------------------------------------------
template <typename Mesh>
void
Teton<Mesh>::setGroups(const std::vector<TetonFreq>& groups) {

    int group = 0,  offset = 0;
    int D_ngr6;

    ngr = groups.size();
    D_ngr6 = std::max(6 * (ngr+1), 0);
   
// Make sure there is room for this group
    gnu.resize(ngr+1);
    quaddef.resize(D_ngr6);

    for (group = 0; group < ngr; ++group) {
        gnu[group]        = groups[group].bot;
        gnu[group+1]      = groups[group].top;
        quaddef[offset]   = groups[group].qtype;
        quaddef[offset+1] = groups[group].qorder;
        quaddef[offset+2] = groups[group].npolar;
        quaddef[offset+3] = groups[group].nazimu;
        quaddef[offset+4] = groups[group].paxis;
        quaddef[offset+5] = -1;     //unused
        offset            = offset + 6;
    }

// Set quadrature definition for acceleration
    quaddef[offset]   = 1;
    quaddef[offset+1] = GTAorder;
    quaddef[offset+2] = 1;
    quaddef[offset+3] = 1;
    quaddef[offset+4] = 1;
    quaddef[offset+5] = -1;// unused

// Construct the quadrature module

    F77_ID(constructquadrature_, constructquadrature, CONSTRUCTQUADRATURE)
        (&ngr, &quaddef[0]);

}

template <typename Mesh>
void Teton<Mesh>::setMeshSizeAndPositions(Teton<Mesh>::MeshType &M,
                                          std::vector<int> &tmpnode) {

    int i = 0, Ti = 0;
    px.resize(ndim*npnts);

    for( i=0; i<npnts; ++i) {
        Ti = tmpnode[i];
        if ( ndim == 1 ) {
            px[i] = (M.nodeBegin() + Ti)->getPosition()[0];
        }
        else if ( ndim == 2 ) {
            px[ndim*i]   = (M.nodeBegin() + Ti)->getPosition()[0];
            px[ndim*i+1] = (M.nodeBegin() + Ti)->getPosition()[1];
        }
        else if ( ndim == 3 ) {
            px[ndim*i]   = (M.nodeBegin() + Ti)->getPosition()[0];
            px[ndim*i+1] = (M.nodeBegin() + Ti)->getPosition()[1];
            px[ndim*i+2] = (M.nodeBegin() + Ti)->getPosition()[2];
        }
    }

}

template <typename Mesh>
void Teton<Mesh>::setCommunication(Teton<Mesh>::MeshType &M,
                                   std::vector<int> &nBdyBC,
                                   std::vector<int> &facetoBC,
                                   std::vector<int> &BdyFaceList,
                                   std::vector<int> &BdyIDList) {

    int i = 0, j = 0, k = 0, l = 0;

    CommMapType  sendMap    = M.getCommAgent().getFaceNeighborMap().sendMap();
    CommMapType  receiveMap = M.getCommAgent().getFaceNeighborMap().receiveMap();
                   
    typename CommMapType::const_iterator iter;
    typename CommMapType::const_iterator Riter;
    typename Teton<Mesh>::MeshType::FaceIterator Fi;
    typename Teton<Mesh>::MeshType::FaceIterator FiR;
    typename Teton<Mesh>::MeshFCiF FaceDomainLookup(M);

    typename Teton<Mesh>::Edge::CornerIterator CorI;

    int faceLID = -1, faceGID = -1, cid = -1, nodeGID = -1;
    int ID1 = -1, maxFace = -1, bcID = -1;
    int CNRfaceCtr = -1, faceCtr = -1, nodeCtr = -1, sharedCtr = -1;
    int neighbor = -1, domainID = -1;
    int maxNode = -1, index = -1, cf = -1;

    int my_node   = M.getCommAgent().getRank();

    std::vector<int> faceGIDList;
    std::vector<int> cnrFace1;
    std::vector<int> nodeGIDList;
    std::vector<int> cornerIDList; 
    std::vector<int> numCNRface;
    std::vector<int> faceLIDList;
    std::vector<int> cidList;

    faceGIDList.reserve( nbshare );
    cnrFace1.reserve( nbshare );
    nodeGIDList.reserve( nbshare );
    cornerIDList.reserve( nbshare );
    numCNRface.reserve( nbshare );
    faceLIDList.reserve( nbshare );
    cidList.resize( nbshare );

    for (Riter = receiveMap.begin(); Riter != receiveMap.end(); ++Riter) {
        domainID = Riter->first;
        for(FiR = Riter->second.begin(); FiR != Riter->second.end(); FiR++) {
            if ( (!(FiR->isExternalSurface()) ) && 
                 (FiR->getOppositeFace().isSend()) ) 
            {
//------------------------------UMT-CMG------------------------------
//  removed getOppositeFace() since global face IDs are the same on different
//  parallel domains.
//------------------------------UMT-CMG------------------------------
//             FaceDomainLookup[FiR->getOppositeFace()] = domainID;
                FaceDomainLookup[*FiR] = domainID;
            }
        }
    }
    
// Iterate over all neighbors 
    sharedCtr = 0;
    
    for (i=0,iter = sendMap.begin(); iter != sendMap.end(); ++iter, i++) {
        CNRfaceCtr = 0;
        faceCtr    = 0;
        neighbor   = iter->first;                 // process ID
        for (Fi = iter->second.begin(); Fi < iter->second.end(); Fi++) {
//      Find all communicate faces
            if ( !(Fi->isExternalSurface() ) && 
                 (Fi->getOppositeFace().isReceive() ) && 
                 (FaceDomainLookup[*Fi] == neighbor) ) 
            {
                faceLID  = M.getLocalID(*Fi);
                bcID     = facetoBC[faceLID];
                if (my_node < neighbor) {
                    faceGID  = M.getGlobalID(*Fi);
                } else if (my_node > neighbor) {
//------------------------------UMT-CMG------------------------------
//  removed getOppositeFace() since global face IDs are the same on different
//  parallel domains and the OppositeFace in CMG for COMM faces is degenerate.
//------------------------------UMT-CMG------------------------------
//               faceGID  = M.getGlobalID(Fi->getOppositeFace());
                    faceGID  = M.getGlobalID(*Fi);
                }

                faceGIDList.push_back( faceGID );
                cnrFace1.push_back( CNRfaceCtr );
                faceLIDList.push_back( faceLID );
                
                nodeCtr = 0;
                for (CorI = Fi->cornerBegin(); CorI < Fi->cornerEnd(); CorI++) {
                    nodeGID = M.getGlobalID(CorI->getNode());
                    cid     = M.getLocalID(*CorI);
                    nodeGIDList.push_back( nodeGID );
                    cornerIDList.push_back( cid );
                    nodeCtr++;
                }
                                                                                                   
                nodeGIDList.resize( nodeCtr );
                cornerIDList.resize( nodeCtr );
                VERIFY2( !nodeGIDList.empty(), "Trying to get the maximum of an empty container.");
                maxNode = (*( std::max_element(nodeGIDList.begin(),nodeGIDList.end()) ));
                                                                                                   
                for (j=0; j<nodeCtr; j++) {
                    index = std::distance( nodeGIDList.begin(), std::min_element(nodeGIDList.begin(),nodeGIDList.end()) );
                    cidList[CNRfaceCtr] = cornerIDList[index];
                    nodeGIDList[index]  = maxNode + 1;
                    CNRfaceCtr++;
                }
                numCNRface[faceCtr]  = nodeCtr;
                faceCtr++;
            }
            nodeGIDList.resize(0);
            cornerIDList.resize(0);
        }
// Now order in increasing face ID and node ID

        if (CNRfaceCtr > 0) {
                                                                                                   
            faceGIDList.resize( faceCtr );
            cnrFace1.resize( faceCtr );
            faceLIDList.resize( faceCtr );
            VERIFY2( !faceGIDList.empty(), "Trying to get the maximum of an empty container.");
            maxFace = (*( std::max_element(faceGIDList.begin(),faceGIDList.end()) ));
             
            for (j=0; j<faceCtr; j++) {
                index = std::distance( faceGIDList.begin(), std::min_element(faceGIDList.begin(),faceGIDList.end()) );
                ID1     = cnrFace1[index];
                faceLID = faceLIDList[index];
                bcID    = facetoBC[faceLID];
                for (k=0; k<numCNRface[index]; k++) {
                    cid                       = cidList[ID1+k];
                    for (l=0;l<maxcf;l++) {
                        if (BdyFaceList[cid*maxcf+l] == faceLID) {
                            cf = l;
                            l  = maxcf;
                        }
                    }
                    BdyIDList[cid*maxcf+cf] = nBdyBC[bcID];
                    nBdyBC[bcID]            = nBdyBC[bcID] + 1;
                }
                faceGIDList[index] = maxFace + 1;
                sharedCtr          = sharedCtr + numCNRface[index];
            }
        }
       
        faceGIDList.resize(0);
        cnrFace1.resize(0);
        faceLIDList.resize(0);
    }

// Release memory
// "resize(0)" does not release memory
    vector<int>().swap(faceGIDList);
    vector<int>().swap(cnrFace1);
    vector<int>().swap(nodeGIDList);
    vector<int>().swap(cornerIDList);
    vector<int>().swap(numCNRface);
    vector<int>().swap(faceLIDList);
    vector<int>().swap(cidList);

}


// ------------------------------------------------------------
// linkKull -- Kull to Teton
//
// Grab stuff from Kull structures, stuff into Teton, and
// ready a step
// ------------------------------------------------------------
template <typename Mesh>
void
Teton<Mesh>::linkKull(Teton<Mesh>::MeshType &M,
                      const std::vector<TetonFreq> &GList,
                      std::vector<TetonBoundary<Mesh> > &BCList,
                      std::vector<TetonVolumeSource> &VSList) {

    int i = 0, j = 0, k = 0, faceno = 0;
    int cid = -1, fid = -1, stride = -1;
    int corner1= -1, corner2 = -1, Ocorner1 = -1;
    int Onode1 = -1, c1 = -1, c2 = -1, Ofid = -1;
    int cfID1 = -1, cfID2 = -1, ez1 = -1, faceCtr = 0;
    int Ocorner2 = -1, Onode2 = -1, node2 = -1;
    int zid = -1, node1 = -1, totfaces = 0;
    int SID = 0, isRecv=0;
    int isTimeDep = 0, isExact = 0, isShared = 0;
    int isRefl = 0, isVac = 0, isTemp = 0, isFDS = 0;
    int isEdit = 0;
    int maxprof = 0, numNodes = 0;
    int nintsides = 0, nodeSize = 0; 
    int numComm = 0, numExt = 0, ncommMax = 0;
    int numBC = 0, numBCTotal = 0, domainID = -1, bcID = -1, bdyElem = -1;
    int offset = 0, last = 0, index = -1;
    int nrefl = 0, nvac = 0, nsrc = 0;
    int bcRefl = 0, bcVac = 0, bcSrc = 0;
    int corner0 = 0, maxCorner = 0, zoneID = 0;

    CharStar8 shared;

    std::vector<int> nBdyBC;
    vector< int > tmpnode, faceMap;
    vector< int > cfaceID1, cfaceID2;
    vector< int > cfaceNext1, cfaceNext2;
    std::vector<int> countN, facetoBC, numCorner;
    std::vector<int> BdyFaceList, BdyIDList;
    std::vector<int> BCProfID, BCDomID, BCEditID, BCNum, BCBdy1;
    std::vector<int> connect, cFaceList, nCPerFace, zoneOpp, gFaceID;
    std::vector<CharStar8> facetoType, BCType;

// Define Iterators
    typename Teton<Mesh>::MeshType::ZoneIterator zidP;
    typename Teton<Mesh>::MeshType::ZoneHandle::CornerIterator cidP;
    typename Teton<Mesh>::MeshType::CornerHandle::SideIterator sidP;
    typename Teton<Mesh>::MeshType::CornerHandle::FaceIterator fidP;

// Determine various problem parameters

    my_node   = M.getCommAgent().getRank();
    n_Procs   = M.getCommAgent().getNumberOfProcessors();
    ncommMax  = M.getCommAgent().getFaceNeighborMap().numberOfSendProcesses();

    maxprof   = std::max(BCList.size(),VSList.size());
    numNodes  = M.getNumberOfNodes();
    nintsides = M.getNumberOfInternalSides();
    nzones    = M.getNumberOfOwnedZones();
    ncornr    = M.getNumberOfOwnedCorners();
    nfaces    = M.getNumberOfOwnedFaces();
    nbshare   = 0;
    ncomm     = 0;
    totfaces  = M.getNumberOfFaces();

    nodeSize   = nintsides;

    strncpy(shared.data,"shared  ",8);

    tmpnode.resize(nintsides);
    PXlookup.resize(numNodes);

    for( i=0; i<nintsides; i++){
        tmpnode[i] = -1;
    }

    for( i=0; i<numNodes; i++){
        PXlookup[i] = -1;
    }

    typename Teton<Mesh>::MeshType::SideIterator Si;

    for(SID=1,Si = M.ownedSideBegin(); Si != M.ownedSideEnd(); Si++,SID++){
        const typename Teton<Mesh>::Zone &Z = Si->getZone();
        isRecv = Z.isReceive();
        if( isRecv ){
            continue;
        }
        tmpnode[SID-1]   = M.getLocalID(Si->getRightCorner().getNode());
    }

    rmdupsort(&tmpnode[0],&nodeSize);
    
    for( i=0; i<nodeSize; i++){
        PXlookup[tmpnode[i]] = i;
    }
    
// Set various mesh parameters
    npnts = nodeSize;

// Find the number of faces per corner (nfpc) and the maximum number (maxcf)
    nfpc.resize(ncornr);
    numCorner.resize(nzones);
    faceMap.resize(totfaces);
    facetoBC.resize(totfaces);
    facetoType.resize(totfaces);

    for( i=0;i<ncornr;i++) {
        nfpc[i] = 0;
    }
    
    for( i=0;i<totfaces;i++) {
        faceMap[i] = 0;
    }

    for( i=0;i<nzones;i++) {
        numCorner[i] = 0;
    }
    
    for(zidP=M.ownedZoneBegin();zidP != M.ownedZoneEnd(); ++zidP) {
        zid = M.getLocalID(*zidP);
        for(cidP=zidP->cornerBegin();cidP != zidP->cornerEnd(); ++cidP) {
            numCorner[zid] = numCorner[zid] + 1;
            corner1 = M.getLocalID(*cidP);
            if (corner1 < ncornr) {
                for(fidP= cidP->faceBegin();fidP != cidP->faceEnd();fidP++) {
                    fid           = M.getLocalID(*fidP);
                    faceMap[fid]  = faceMap[fid]  + 1;
                    nfpc[corner1] = nfpc[corner1] + 1;
                }
            }
        }
    }
    
    VERIFY2( !nfpc.empty(), "Trying to get the maximum element of an empty container.");
    maxcf = (*( std::max_element(nfpc.begin(),nfpc.end()) ));

    VERIFY2( !numCorner.empty(), "Trying to get the maximum element of an empty container.");
    maxCorner = (*( std::max_element(numCorner.begin(),numCorner.end()) ));
    
    faceCtr = 0;
    for( i=0;i<totfaces;i++) {
        if (faceMap[i] > 0) {
            faceCtr++;
            faceMap[i] = faceCtr;
        }
    }
    
    // YKT experiment:
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    //cudaFuncCache cachesetting = cudaFuncCachePreferShared;
    //cudaDeviceSetCacheConfig(cachesetting);


// Allocate persistant arrays
    resize();

    F77_ID(constructprofile_, constructprofile, CONSTRUCTPROFILE)
        (&maxprof);
    
    setBCs(BCList);
    setVSs(VSList);

    CommMapType  receiveMap = M.getCommAgent().getFaceNeighborMap().receiveMap();
    typename CommMapType::const_iterator Riter;
    typename Teton<Mesh>::MeshType::FaceIterator FiR;
    typename Teton<Mesh>::MeshFCiF FaceDomainLookup(M);

    countN.resize(n_Procs);

    BdyFaceList.resize(ncornr*maxcf);  // local face ID
    BdyIDList.resize(ncornr*maxcf);    // boundary element ID

    for ( i=0;i<ncornr*maxcf;i++) {
        BdyFaceList[i] = -1;
        BdyIDList[i]   = -1;
    }

    for ( i=0;i<n_Procs;i++) {
        countN[i] = 0;
    }

    for (Riter = receiveMap.begin(); Riter != receiveMap.end(); ++Riter) {
        domainID = Riter->first;
        for(FiR = Riter->second.begin(); FiR != Riter->second.end(); FiR++) {
            if ( !(FiR->isExternalSurface()) && 
                 (FiR->getOppositeFace().isSend()) )
            {
//------------------------------UMT-CMG------------------------------
//  removed getOppositeFace() since global face IDs are the same on different
//  parallel domains and the OppositeFace in CMG for COMM faces is degenerate.
//------------------------------UMT-CMG------------------------------
//             FaceDomainLookup[FiR->getOppositeFace()] = domainID;
                FaceDomainLookup[*FiR] = domainID;
            }
            else
            {
                cout<<" WHOA!  receive map face may be external surface or opposite is not a send face.  Teton.cc L869"<<endl;
            }
            
        }
    }
    
// -------------------------------------------------------------------
// Go through the BC's and build up Teton BC arrays Based on Corners
// -------------------------------------------------------------------

    for( i=0; i<nbc; ++i) {
        TetonBoundary<Mesh> &BC = BCList[i];

        isRefl = BC.profile.typeIs("refl");
        isVac  = BC.profile.typeIs("vac");
        isTemp = BC.profile.typeIs("temp");
        isFDS  = BC.profile.typeIs("fds");
        isEdit = BC.profile.typeIs("edit");

        if ( isEdit ) {
            if (my_node == 0) {
                printf("*** Deprecation Warning: edit boundary conditions have been deprecated \n");
                printf("and can be removed from the radiation bc list.  Edits for all source \n");
                printf("and vacuum boundaries are now computed by default. *** \n");
            }
            continue;
        }

        if (BC.faceids.size() > 0) {

            if ( isRefl ) {
                nrefl++;
            } else if ( isVac ) {
                nvac++;
            } else if ( isTemp ) {
                nsrc++;
            } else if ( isFDS ) {
                nsrc++;
            } else {
                ASSERT2(false, "Invalid radiation boundary condition.");
            }
        }
    }

    numBCTotal = nrefl + nvac + nsrc + ncommMax;
    
    
    nBdyBC.resize(numBCTotal);
    BCNum.resize(numBCTotal);
    BCBdy1.resize(numBCTotal);
    BCType.resize(numBCTotal);
    BCProfID.resize(numBCTotal);
    BCDomID.resize(numBCTotal);
    BCEditID.resize(numBCTotal);
    zonetosrc.resize(nzones);

    for ( i=0;i<numBCTotal;i++) {
        nBdyBC[i]   =  0;
        BCNum[i]    =  0;
        BCBdy1[i]   =  0;
        BCProfID[i] = -1;
        BCDomID[i]  = -1;
        BCEditID[i] = -1;
    }

// We order the boundaries by type: refl, vac, src and shared
    
    numBC  = nrefl + nvac + nsrc;
    bcSrc  = nrefl + nvac;
    bcVac  = nrefl;
    bcRefl = 0;
    
    nbedit = 0;
    for( i=0; i<nbc; ++i) {
        
        TetonBoundary<Mesh> &BC = BCList[i];

        isRefl = BC.profile.typeIs("refl");
        isVac  = BC.profile.typeIs("vac");
        isEdit = BC.profile.typeIs("edit");

        if ( isEdit ) {
            continue;
        }

// EditID is global
        if ( isRefl ) {
        } else  {
            nbedit++;
        }

        if (BC.faceids.size() > 0) {

            if ( isRefl ) {
                bcID = bcRefl;
                bcRefl++;
                
            } else if ( isVac ) {
                bcID = bcVac;
                BCEditID[bcID] = nbedit;
                bcVac++;
            } else  {
                bcID = bcSrc;
                BCEditID[bcID] = nbedit;
                bcSrc++;
            }

            BCType[bcID]   = BC.profile.type;
            BCProfID[bcID] = BC.profile.profID;
            
            for( faceno=0; faceno < BC.faceids.size(); ++faceno) {
                fid               = BC.faceids[faceno];
                facetoBC[fid]     = bcID;
                facetoType[fid]   = BC.profile.type;
            }
            
        }
        
    }
    
// -------------------------------------------------------------------
// Go through the VS's and build up Teton VS array (ZoneToSrc)
// -------------------------------------------------------------------
    
    // Initialize all elements of 'zone ID' to 'VSP ID' mapping to zero
    // (NULL volume source profile)
    for ( i=0; i<nzones; ++i ) zonetosrc[i] = 0;

    // Loop over all volume source profiles
    // NOTE: We start the VSP index at one since zero refers to a NULL
    // volume source profile (convention defined by FORTRAN code)
    for ( i=1; i<=nvs; ++i ) {
        const TetonVolumeSource &TVS = VSList[i-1];
        
        // Loop over all zones associated with i-th volume source profile
        for ( j=0; j<TVS.zoneids.size(); ++j ) {
            // Set VSP associated with zone
            zonetosrc[TVS.zoneids[j]] = i;
        }
    }

// Communicate boundaries

    numComm = 0;

    for(zidP=M.ownedZoneBegin();zidP != M.ownedZoneEnd(); ++zidP) {
        for(fidP= zidP->faceBegin();fidP != zidP->faceEnd();fidP++) {

            if( (fidP->isSend()) && !(fidP->isExternalSurface()) && 
                (fidP->getOppositeFace().isReceive()) ) 
            {
                domainID        = FaceDomainLookup[*fidP];
                fid             = M.getLocalID(*fidP);
                facetoType[fid] = shared;
                facetoBC[fid]   = domainID;


                for(cidP=fidP->cornerBegin();cidP != fidP->cornerEnd(); ++cidP) {
                    numComm++;
                    countN[domainID] = countN[domainID] + 1;
                }

            } else if(M.isExternal(*fidP)) {
                ASSERT2(false, "Should never have external faces for a mesh.");
//          } else if(M.isInternal(*fidP) && fidP->getOppositeFace().isExternalSurface()) {
//             } else if(M.isInternal(*fidP) && fidP->isExternalSurface()) {
            } else if(fidP->isExternalSurface()) {
                fid  = M.getLocalID(*fidP);
                bcID = facetoBC[fid];
                for(cidP=fidP->cornerBegin();cidP != fidP->cornerEnd(); ++cidP) {
                    numExt++;
                    nBdyBC[bcID] = nBdyBC[bcID] + 1;
                }
                
            }
        }
    }

    ncomm = 0;
    for ( i=0;i<n_Procs;i++) {
        if (countN[i] > 0) {
            nBdyBC[numBC+ncomm]  = countN[i];
            countN[i]            = numBC + ncomm;
            BCType[numBC+ncomm]  = shared;
            BCDomID[numBC+ncomm] = i;
            ncomm++;
        }
    }

    for( fid=0;fid<totfaces;fid++) {
        isShared = (strncmp(facetoType[fid].data, "shared  ",8) ==0);
        if (isShared) {
            domainID      = facetoBC[fid];
            facetoBC[fid] = countN[domainID];
        }
    }

    nbelem     = numExt + numComm;
    nbshare    = numComm;
    numBCTotal = numBC + ncomm;

    nBdyBC.resize(numBCTotal);

    offset = 0;
    for ( i=0;i<numBCTotal;i++) {
        BCNum[i]  = nBdyBC[i];
        BCBdy1[i] = offset + 1;
        last      = nBdyBC[i];
        nBdyBC[i] = offset;
        offset    = offset + last; 
    }

    F77_ID(constructboundary_, constructboundary, CONSTRUCTBOUNDARY)
        (&nrefl, &nvac, &nsrc, &ncomm);

// Set Bdy Lists
    for(zidP=M.ownedZoneBegin();zidP != M.ownedZoneEnd(); ++zidP) {
        for(cidP=zidP->cornerBegin();cidP != zidP->cornerEnd(); ++cidP) {
            cid   = M.getLocalID(*cidP);
            index = 0;
            for(fidP= cidP->faceBegin();fidP != cidP->faceEnd();fidP++) {
                if( (fidP->isSend()) && !(fidP->isExternalSurface()) && 
                    (fidP->getOppositeFace().isReceive()) ) {
//              Communicate Faces are done in setCommunication
                    fid                          = M.getLocalID(*fidP);
                    BdyFaceList[cid*maxcf+index] = fid;
                    index++;
                } else if(M.isExternal(*fidP)) {
                    ASSERT2(false, "Should never have external faces for a mesh.");
//             } else if(M.isInternal(*fidP) && fidP->getOppositeFace().isExternalSurface()) {
//             } else if(M.isInternal(*fidP) && fidP->isExternalSurface()) {
                } else if(fidP->isExternalSurface()) {
                 
                    fid                          = M.getLocalID(*fidP);
                    bcID                         = facetoBC[fid];
                    BdyFaceList[cid*maxcf+index] = fid;
                    BdyIDList[cid*maxcf+index]   = nBdyBC[bcID];
                    nBdyBC[bcID]                 = nBdyBC[bcID] + 1;
                    index++;
                }
            }
        }
    }

// Set communication information for parallel runs (BdyFaceList, BdyIDList)
                                                                                                   
    if (n_Procs > 1) {
        strncpy(decomp_s.data,"on      ",8);
        setCommunication(M, nBdyBC, facetoBC, BdyFaceList, BdyIDList);
    }

// Set the node positions
    setMeshSizeAndPositions(M, tmpnode);

// Construct Teton Modules
    setGroups(GList);
   
    double dummyRadForceMultiplier = 0.0;
    
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >=1100 )
    long int stringSize=8;
    F77_ID(constructsize_, constructsize, CONSTRUCTSIZE)
        (&my_node, &nzones, &ncornr, &nfaces, &npnts,
         &nbelem, &ndim, &maxcf, &maxCorner, &ngr, 
         &nangsn, &npsi, &ncomm, &nbshare, &nbedit,
         &tfloor, &tmin, &dummyRadForceMultiplier,
         igeom.data, ittyp.data, iaccel.data,
         iscat.data, itimsrc.data, decomp_s.data,
         &stringSize,&stringSize,&stringSize,
         &stringSize,&stringSize,&stringSize);
#else
    F77_ID(constructsize_, constructsize, CONSTRUCTSIZE)
        (&my_node, &nzones, &ncornr, &nfaces, &npnts,
         &nbelem, &ndim, &maxcf, &maxCorner, &ngr, 
         &nangsn, &npsi, &ncomm, &nbshare, &nbedit,
         &tfloor, &tmin, &dummyRadForceMultiplier,
         igeom.data, ittyp.data, iaccel.data,
         iscat.data, itimsrc.data, decomp_s.data);
#endif

// Set the Quadrature module and resize arrays
    F77_ID(setsnorder_, setsnorder, SETSNORDER)
        (&nangsn, &quaddef[0], &gnu[0]);

    isTimeDep = ( strncmp(ittyp.data,  "timedep ",8) == 0);
    isExact   = ( strncmp(itimsrc.data,"exact   ",8) == 0);
    if ( isTimeDep && isExact ) {
        npsi = nangsn*ngr;
    }
    else {
        npsi = 1;
    }

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >=1100 )
    F77_ID(addboundary_, addboundary, ADDBOUNDARY)
        (&numBCTotal, BCType[0].data, &BCNum[0],
         &BCBdy1[0], &BCProfID[0], &BCDomID[0], &BCEditID[0],&stringSize);
#else
    F77_ID(addboundary_, addboundary, ADDBOUNDARY)
        (&numBCTotal, BCType[0].data, &BCNum[0],
         &BCBdy1[0], &BCProfID[0], &BCDomID[0], &BCEditID[0]);
#endif                                                                                          
    resize();

    F77_ID(constructmaterial_, constructmaterial, CONSTRUCTMATERIAL)
        ();

    F77_ID(constructeditor_, constructeditor, CONSTRUCTEDITOR)
        ();

    F77_ID(constructgeometry_, constructgeometry, CONSTRUCTGEOMETRY)
        ();

   
// Allocate arrays to construct the mesh connectivity
    connect.resize(3*maxcf*maxCorner);
    cFaceList.resize(maxCorner);
    nCPerFace.resize(maxCorner);
    zoneOpp.resize(maxCorner);
    gFaceID.resize(maxCorner);
   
    ctozone.resize(ncornr);
    ctopoint.resize(ncornr);
    bdytoc.resize(nbelem);
    cfaceNext1.resize(ncornr);
    cfaceNext2.resize(ncornr);
    cfaceID1.resize(maxcf*ncornr);
    cfaceID2.resize(maxcf*ncornr);
    ctoface.resize(maxcf*ncornr);

// For now, the building of connect is geometry specific. Hopefully,
// it can be generalized in the near future

    stride = 3*maxcf;


    for( i=0;i<ncornr;i++) {
        cfaceNext1[i] = 0;
        cfaceNext2[i] = 0;
    }

    for(zidP=M.ownedZoneBegin();zidP != M.ownedZoneEnd(); ++zidP) {
        for(sidP= zidP->sideBegin();sidP != zidP->sideEnd();sidP++) {
           
            corner1  = M.getLocalID(sidP->getRightCorner());
            corner2  = M.getLocalID(sidP->getLeftCorner());
         
            if ( (corner1 < ncornr) && (corner2 < ncornr) ) {
                fid = M.getLocalID(sidP->getFace());
                j   = cfaceNext2[corner1];
                k   = cfaceNext1[corner2];

                ctoface[corner2*maxcf+k]  = fid;
                cfaceID1[corner2*maxcf+k] = corner1;
                cfaceID2[corner1*maxcf+j] = corner2;
                cfaceNext1[corner2]       = cfaceNext1[corner2] + 1;
                cfaceNext2[corner1]       = cfaceNext2[corner1] + 1;
            } else {
            }
        }
    }

    for( i=0;i<ncornr;i++) {
        for( j=1;j<maxcf-1;j++) {
            cfID1 = cfaceID1[i*maxcf+j];
            cfID2 = cfaceID2[i*maxcf+j];
            fid   = ctoface[i*maxcf+j];
            if (cfID1 != cfaceID2[i*maxcf+j-1]) {
                cfaceID1[i*maxcf+j]   = cfaceID1[i*maxcf+j+1];
                cfaceID2[i*maxcf+j]   = cfaceID2[i*maxcf+j+1];
                ctoface[i*maxcf+j]    = ctoface[i*maxcf+j+1];
                cfaceID1[i*maxcf+j+1] = cfID1;
                cfaceID2[i*maxcf+j+1] = cfID2;
                ctoface[i*maxcf+j+1]  = fid;
            }
        }
    }

    corner0 = 0;
    bdyElem = 0;
    for(zidP=M.ownedZoneBegin();zidP != M.ownedZoneEnd(); ++zidP) {
        zid = M.getLocalID(*zidP);
        faceCtr = 0;
        for(fidP= zidP->faceBegin();fidP != zidP->faceEnd();fidP++) {
            fid = M.getLocalID(*fidP);
            for(sidP= fidP->sideBegin();sidP != fidP->sideEnd();sidP++) {
                corner1  = M.getLocalID(sidP->getRightCorner());
                Ocorner1 = M.getLocalID(sidP->getOppositeSide().getLeftCorner());
                Onode1   = M.getLocalID(sidP->getOppositeSide().getLeftCorner().getNode());
                node1    = M.getLocalID(sidP->getRightCorner().getNode());
                Ofid     = M.getLocalID(sidP->getOppositeSide().getFace());

                bool isThisSidesFaceInternal = fidP->isInternal();

                c1       = (corner1 - corner0)*stride;
 
                corner2=-1;
                for( j=0;j<maxcf;j++) {
                    if (ctoface[corner1*maxcf + j] == fid) {
                        cfID1   = j;
                        corner2 = cfaceID1[corner1*maxcf + j];
                    }
                }
                ASSERT( corner2 != -1 );
                
                for( j=0;j<maxcf;j++) {
                    if (cfaceID1[corner2*maxcf + j] == corner1) {
                        cfID2 = j;
                    }
                }

                ctopoint[corner1] = PXlookup[node1] + 1;
                ctozone[corner1]  = zid + 1;

// Set the "EZ" corner faces shared by corner1/corner2
                ez1 = 3*cfID1 + 2;

                connect[c1 + ez1] = corner2 + 1;

// Set the "FP" corner faces
                //
                // CMG_UMT: This checks whether the opposite corner is internal
                // We don't have external corners in CMG so we'll replace this 
                // with a check for is this corner's face external surface
                //
                // Note that the original code had if(Onode1 == node1) condition
                // outside of if(Ocorner1 < ncornr).  But for external surface 
                // faces, the opposite node will be -1.
                //
//                if (Ocorner1 < ncornr) {
                if( isThisSidesFaceInternal )
                {
                    if (Onode1 == node1) {

                        for( j=0;j<maxcf;j++) {
                            if (ctoface[Ocorner1*maxcf + j] == Ofid) {
                                cfID2 = j;
                            }
                        }
                        
                        connect[c1  + 3*cfID1]     = Ocorner1 + 1;
                        connect[c1  + 3*cfID1 + 1] = cfID2 + 1;
                    }// end of check for identical node ids for opposite nodes.
                } else {
                    for( j=0;j<maxcf;j++) {
                        if (fid == BdyFaceList[corner1*maxcf+j]) {
                            bdyElem = BdyIDList[corner1*maxcf+j];
                            break;
                        }
                    }
                        
                    bdytoc[bdyElem]           = corner1 + 1;
                    connect[c1 + 3*cfID1]     = 0;
                    connect[c1 + 3*cfID1 + 1] = bdyElem + 1;
                }// end of is this side's face internal condition
            }//end of loop over sides
            faceCtr++;
        }//end of loop over faces

        zoneID = zid + 1;
        F77_ID(setzone_, setzone, SETZONE)
            (&zoneID, &corner0, &faceCtr, &numCorner[0], &connect[0], &nfpc[0]);

        corner0 = corner0 + numCorner[zid];
    }//end of loop over zones

// The Sn package expects the faces to be numbered 1 to nfaces.
    for(i=0; i<maxcf*ncornr; ++i) {
        j          = ctoface[i];
        fid        = faceMap[j];
        ctoface[i] = fid;
    }


    F77_ID(setgeometry_, setgeometry, SETGEOMETRY)
        (&ctozone[0], &ctopoint[0], &ctoface[0], 
         &nfpc[0], &bdytoc[0], &zonetosrc[0], &px[0]); 

// Release temporary variables
// "resize(0)" does not release memory
    std::vector<int>().swap(tmpnode);
    std::vector<int>().swap(nfpc);
    std::vector<int>().swap(numCorner);
    std::vector<int>().swap(faceMap);
    std::vector<int>().swap(connect);
    std::vector<int>().swap(cFaceList);
    std::vector<int>().swap(nCPerFace);
    std::vector<int>().swap(zoneOpp);
    std::vector<int>().swap(gFaceID);
    std::vector<int>().swap(ctozone);
    std::vector<int>().swap(ctopoint);
    std::vector<int>().swap(bdytoc);
    std::vector<int>().swap(cfaceNext1);
    std::vector<int>().swap(cfaceNext2);
    std::vector<int>().swap(cfaceID1);
    std::vector<int>().swap(cfaceID2);
    std::vector<int>().swap(ctoface);
    std::vector<int>().swap(zonetosrc);
   

    /*cout<<"pinning psi and phi in C code"<<endl;
    // Pin the memory (psi should be pinned when created in kull, not in Teton/radtr)
    F77_ID(pinmem_, pinmem, PINMEM)
      ( &psir[0],               // double *
        &Phi[0] );               // double *
    */



}

template <typename Mesh>
void
Teton<Mesh>::CInitMaterial(PartList<Mesh> &partList)
{

    int  nZones, zid, ownedZones;
    double Elocal;

    mPartListPtr   = &partList;
    const Mesh &M  = mPartListPtr->getMesh();

    ownedZones     = M.getNumberOfOwnedZones(); 
    EnergyMaterial = 0.0;

// Initialize arrays to handle multi-material zones
    for(zid=0; zid<ownedZones; zid++){
        rho[zid] = 0.0;
        cve[zid] = 0.0;
        tez[zid] = 0.0;
        trz[zid] = 0.0;
    }

    typename PartList<Mesh>::Iterator partPtr;

    for(partPtr =  mPartListPtr->begin(); partPtr !=  mPartListPtr->end(); partPtr++) {
        const Region<Mesh> &r = partPtr->getRegion();
        const MeshType     &m = r.getMesh();
        typename Region<Mesh>::ZoneIterator zidP = r.ownedZoneBegin();
        nZones = r.getNumberOfOwnedZones();
        if(nZones == 0){
            continue;
        }
//      partPtr->setElectronSpecificHeatCV ();
//      const RZCSF &CV = partPtr->getElectronSpecificHeat ();
        partPtr->setSpecificHeatCV ();
        const RZCSF &CV = partPtr->getSpecificHeatCV ();
        const RZCSF &MD = partPtr->getMassDensity ();
        const RZCSF &ET = partPtr->getElectronTemperature ();
        const RZCSF &RT = partPtr->getRadiationTemperature ();
        const RZCSF &SE = partPtr->getSpecificEnergy ();
        for(; zidP != r.ownedZoneEnd(); ++zidP) {
            zid             = m.getLocalID(*zidP);
            rho[zid]       += MD[*zidP];
            cve[zid]       += MD[*zidP]*CV[*zidP];
            tez[zid]       += MD[*zidP]*CV[*zidP]*ET[*zidP];
            trz[zid]       += MD[*zidP]*RT[*zidP];
            EnergyMaterial += zidP->getVolume()*MD[*zidP]*SE[*zidP];
        }
    }

// Check the number of groups for the opacity

    Elocal = EnergyMaterial;
    MPI_Allreduce(&Elocal, &EnergyMaterial, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
       
    for(zid=0; zid<ownedZones; zid++){
        tez[zid]     = tez[zid]/cve[zid];
        cve[zid]     = cve[zid]/rho[zid];
        trz[zid]     = trz[zid]/rho[zid];
        denez[zid]   = 0.0;
        SMatEff[zid] = 0.0;
    }

    F77_ID(setmaterialmodule_, setmaterialmodule, SETMATERIALMODULE)
        (&siga[0], &sigs[0], &cve[0], &rho[0],
         &SMatEff[0], &denez[0], &trz[0], &tez[0], &tec[0]); 

    F77_ID(seteditormodule_, seteditormodule, SETEDITORMODULE)
        (&RadEnergyEscRate[0], &RadEnergyIncRate[0],
         &RadEnergyEscape[0], &RadEnergyIncident[0]);

    F77_ID(getgeometry_, getgeometry, GETGEOMETRY)();

// Initialize corner radiation variables

    //cout<<"calling rtinit. ownedZones = "<<ownedZones<<endl;

    F77_ID(rtinit_, rtinit, RTINIT)
        (&EnergyRadiation, &psir[0]);

    //cout<<"finished rtinit"<<endl;

    F77_ID(setenergyedits_, setenergyedits, SETENERGYEDITS)
        (&EnergyRadiation, &EnergyMaterial, &EnergyIncident, 
         &EnergyEscaped, &EnergyExtSources,
         &EnergyCheck, &deltaEsrc, &deltaEhyd);

}

template <typename Mesh>
void
Teton<Mesh>::UpdateOpacity () {

    int  ig, nZones, zid, rzid, ownedZones;
    int isScat = 0;
    std::vector<double> opacA, opacS, rhoH, tempE, tempR;
   
    isScat = ( strncmp(iscat.data,   "on      ", 8) == 0);

    ownedZones = rho.size();
                                                                                                   
// Initialize opacities to handle multi-material zones
    for(zid=0; zid<ownedZones; zid++){
        for(ig=0; ig<ngr; ig++){
            siga[zid*ngr + ig] = 0.0;
            sigs[zid*ngr + ig] = 0.0;
        }
    }
                                                                                                   
    typename PartList<Mesh>::Iterator partPtr;
    for(partPtr =  mPartListPtr->begin(); partPtr !=  mPartListPtr->end(); partPtr++) {
        const Region<Mesh> &r = partPtr->getRegion();
        const MeshType     &m = r.getMesh();
        typename Region<Mesh>::ZoneIterator zidP = r.internalZoneBegin();
        nZones = r.getNumberOfZones();
        if(nZones == 0){
            continue;
        }
        opacA.resize(nZones*ngr);
        opacS.resize(nZones*ngr);
        rhoH.resize(nZones);
        tempE.resize(nZones);
        tempR.resize(nZones);
      
        const RZCSF &MD = partPtr->getMassDensity ();
        const RZCSF &ET = partPtr->getElectronTemperature ();
        const RZCSF &RT = partPtr->getRadiationTemperature ();

        for(; zidP != r.internalZoneEnd(); ++zidP) 
        {
            zid  = m.getLocalID(*zidP);
            rzid = r.getLocalID(*zidP);
         
            rhoH[rzid]  = MD[*zidP];
            tempE[rzid] = ET[*zidP];
            tempR[rzid] = RT[*zidP];
         
            for(ig=0; ig < ngr; ig++){
                opacS[rzid*ngr +ig] = 0.0;
            }
        }
                                                                                                 
        partPtr->getMaterial().getOpacity().getAbsorption(opacA, rhoH, tempE);

        if ( isScat ) {
            partPtr->getMaterial().getOpacity().getScattering(opacS, rhoH, tempE, tempR);
        }
              
        for(zidP=r.ownedZoneBegin(); zidP != r.ownedZoneEnd(); ++zidP) {
            zid = m.getLocalID(*zidP);
            rzid = r.getLocalID(*zidP);
            if( zid < ownedZones )
            {
                for(ig=0; ig < ngr; ig++){
                    siga[zid*ngr +ig] += MD[*zidP]*opacA[rzid*ngr +ig];
                    sigs[zid*ngr +ig] += MD[*zidP]*opacS[rzid*ngr +ig];
                }
            }
        }
          
    }
          
}


template <typename Mesh>
void
Teton<Mesh>::CupdateSn () {

    int  nZones, zid, ownedZones;
    double Elocal;
    const double cvefloor=0.001;

    typename Teton<Mesh>::MeshType::ZoneIterator zidP;

    const Mesh &M = mPartListPtr->getMesh();

    ownedZones     = M.getNumberOfOwnedZones();
    EnergyMaterial = 0.0;

// Initialize state variables to handle multi-material zones
    for(zid=0; zid<ownedZones; zid++){
        rho[zid] = 0.0;
        cve[zid] = 0.0;
        tez[zid] = 0.0;
    }

    typename PartList<Mesh>::Iterator partPtr;

    for(partPtr =  mPartListPtr->begin(); partPtr !=  mPartListPtr->end(); partPtr++) {
        const Region<Mesh> &r = partPtr->getRegion();
        const MeshType     &m = r.getMesh();
        typename Region<Mesh>::ZoneIterator zidP = r.internalZoneBegin();
        nZones = r.getNumberOfZones();
        if(nZones == 0){
            continue;
        }
//      const RZCSF &CV = partPtr->state().get(Rad3TCommon<Mesh>::EFFECTIVE_CV());
        partPtr->setSpecificHeatCV ();
        const RZCSF &CV = partPtr->getSpecificHeatCV ();
        const RZCSF &MD = partPtr->getMassDensity ();
        const RZCSF &SE = partPtr->getSpecificEnergy ();
        const RZCSF &ET = partPtr->getElectronTemperature ();

        for(; zidP != r.internalZoneEnd(); ++zidP) {
            zid  = m.getLocalID(*zidP);
            if (zid < ownedZones) {
                rho[zid]       += MD[*zidP];
                cve[zid]       += MD[*zidP]*CV[*zidP];
                tez[zid]       += MD[*zidP]*CV[*zidP]*ET[*zidP];
                EnergyMaterial += MD[*zidP]*SE[*zidP]*zidP->getVolume();
            }
        }
    }

    Elocal = EnergyMaterial;
    MPI_Allreduce(&Elocal, &EnergyMaterial, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for(zid=0; zid<ownedZones; zid++){
        tez[zid] = tez[zid]/cve[zid];
        cve[zid] = cve[zid]/rho[zid];
        cve[zid] = std::max(cve[zid], cvefloor);
    }

}

template <typename Mesh>
void
Teton<Mesh>::CupdateSpecificEnergy(std::vector<double> &deltaEnergy,
                                   MeshType &psurface,
                                   PartList<Mesh>  &realPartList) {

   int  zid, nZones, ownedZones;
   double deltaSE, totalME, delME, oldSE, frac, VFmult;
   std::vector<double> tempSE, Edep;

   PartList<Mesh> *partList;
   partList = &realPartList;

   totalME      = 0.0;
   delME        = 0.0;

   ownedZones = psurface.getNumberOfOwnedZones();
   tempSE.resize(ownedZones);
   Edep.resize(ownedZones);

   for(zid=0; zid<ownedZones; zid++){
      tempSE[zid] = 0.0;
      Edep[zid]   = rho[zid]*(denez[zid] - deltaEnergy[zid]);
   }

   typename PartList<Mesh>::Iterator partPtr;
   for(partPtr =  partList->begin(); partPtr !=  partList->end(); partPtr++) {
      const Region<Mesh> &r = partPtr->getRegion();
      const MeshType     &m = r.getMesh();
      typename Region<Mesh>::ZoneIterator zidP =r.internalZoneBegin();
      nZones = r.getNumberOfZones();
      if(nZones == 0){
         continue;
      }
      typename Part<Mesh>::ZonalScalarFieldType &SH = partPtr->getSpecificHeatCV ();
      typename Part<Mesh>::ZonalScalarFieldType &T  = partPtr->getTemperature ();
      typename Part<Mesh>::ZonalScalarFieldType &MD = partPtr->getMassDensity ();
      typename Part<Mesh>::ZonalScalarFieldType &VF = partPtr->getVolumeFraction ();
      for(; zidP != r.internalZoneEnd(); ++zidP) {
         zid         = m.getLocalID(*zidP);
         if (Edep[zid] < 0.0) {
            VFmult = 0.5*(1.0 + VF[*zidP]);
         }
         else {
            VFmult = 1.0;
         }
         tempSE[zid] = tempSE[zid] + VFmult*MD[*zidP]*SH[*zidP]*T[*zidP];
      }
   }

   for(partPtr =  partList->begin(); partPtr !=  partList->end(); partPtr++) {
      const Region<Mesh> &r = partPtr->getRegion();
      const MeshType     &m = r.getMesh();
      typename Region<Mesh>::ZoneIterator zidP =r.internalZoneBegin();
      nZones = r.getNumberOfZones();
      if(nZones == 0){
         continue;
      }
      typename Part<Mesh>::ZonalScalarFieldType &SH = partPtr->getSpecificHeatCV ();
      typename Part<Mesh>::ZonalScalarFieldType &T  = partPtr->getTemperature ();
      typename Part<Mesh>::ZonalScalarFieldType &SE = partPtr->getSpecificEnergy ();
      typename Part<Mesh>::ZonalScalarFieldType &MD = partPtr->getMassDensity ();
      typename Part<Mesh>::ZonalScalarFieldType &ET = partPtr->getElectronTemperature ();
      typename Part<Mesh>::ZonalScalarFieldType &RT = partPtr->getRadiationTemperature ();
      typename Part<Mesh>::ZonalScalarFieldType &VF = partPtr->getVolumeFraction ();
      for(; zidP != r.internalZoneEnd(); ++zidP) {
         zid       = m.getLocalID(*zidP);
         if (Edep[zid] < 0.0) {
            VFmult = 0.5*(1.0 + VF[*zidP]);
         }
         else {
            VFmult = 1.0;
         }
         deltaSE   = VFmult*SH[*zidP]*T[*zidP]*Edep[zid]/tempSE[zid];
         SE[*zidP] = SE[*zidP] + deltaSE;

         totalME   = totalME + zidP->getVolume()*MD[*zidP]*SE[*zidP];
         delME     = delME   + zidP->getVolume()*MD[*zidP]*deltaSE;
         RT[*zidP] = trz[zid];
      }

      partPtr->setTemperature();

      for(zidP=r.internalZoneBegin(); zidP != r.internalZoneEnd(); ++zidP) {
         zid           = m.getLocalID(*zidP);
          T[*zidP]     = std::max(T[*zidP], tfloor);
         ET[*zidP]     = T[*zidP];
      }
        
      partPtr->setPressure();
      partPtr->setSoundSpeed();
      partPtr->setGamma();
   }

}


template <typename Mesh>
void
Teton<Mesh>::CrelinkMesh() {

    int nid,  pxNid;
    typename Teton<Mesh>::MeshType::NodeIterator nidP;

    const Mesh &M = mPartListPtr->getMesh();
                                                                                        
    for(nidP = M.nodeBegin();nidP != M.nodeEnd(); nidP++) {
        nid   = M.getLocalID(*nidP);
        pxNid = PXlookup[nid];
        if (pxNid < npnts) {
            if (pxNid > -1) {
                if (ndim == 3) {
                    px[pxNid*3]     = nidP->getPosition()[0];
                    px[pxNid*3 + 1] = nidP->getPosition()[1];
                    px[pxNid*3 + 2] = nidP->getPosition()[2];
                } else if ( ndim == 2 ) {
                    px[pxNid*2]     = nidP->getPosition()[0];
                    px[pxNid*2 + 1] = nidP->getPosition()[1];
                } else if ( ndim == 1 ) {
                    px[pxNid]       = nidP->getPosition()[0];
                }
            }
        }
    }

    Timer_Beg("getgeometry");
    nvtxRangePushA("getgeometry");
    F77_ID(getgeometry_, getgeometry, GETGEOMETRY)
        ();
    nvtxRangePop();
    Timer_End("getgeometry");
      
}

template <typename Mesh>
void
Teton<Mesh>::CsetControls() {
    
    double dummyRadForceMultiplier=0.0;
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >=1100 )
    long int stringSize=8;
    
    F77_ID(resetsize_, resetsize, RESETSIZE)
        (&tfloor, &tmin, &dummyRadForceMultiplier, igeom.data, ittyp.data, 
         iaccel.data, iscat.data, itimsrc.data, decomp_s.data,
         &stringSize,&stringSize,&stringSize,
         &stringSize,&stringSize,&stringSize);
#else
    F77_ID(resetsize_, resetsize, RESETSIZE)
        (&tfloor, &tmin, &dummyRadForceMultiplier, igeom.data, ittyp.data, 
         iaccel.data, iscat.data, itimsrc.data, decomp_s.data);
#endif

    F77_ID(constructitercontrols_, constructitercontrols, CONSTRUCTITERCONTROLS)
        (&noutmx, &ninmx, &ngdamx,
         &epstmp, &epsinr, &epsgda);

    F77_ID(constructdtcontrols_, constructdtcontrols, CONSTRUCTDTCONTROLS)
        (&dtrad, &dtrmn, &dtrmx, &delte, &deltr);

}

template <typename Mesh>
void
Teton<Mesh>::getBoundaryEdits() {

    int editID, group, n;
    double deltaGnu;
    double eps=1.0e-80;
                                                                                                     
    for (editID=0; editID<nbedit; editID++) {
        RE_Escape[editID]       = 0.0;
        RE_Incident[editID]     = 0.0;
        RE_EscapeRate[editID]   = 0.0;
        RE_IncidentRate[editID] = 0.0;

        n = editID*ngr;
        for (group=0; group<ngr; group++) {
            deltaGnu = gnu[group+1] - gnu[group];
            RE_Escape[editID]               += RadEnergyEscape[n+group];
            RE_Incident[editID]             += RadEnergyIncident[n+group];
            RE_EscapeRate[editID]           += RadEnergyEscRate[n+group];
            RE_IncidentRate[editID]         += RadEnergyIncRate[n+group];

            RE_EscapeSpectrum[n+group]       = (RadEnergyEscape[n+group]+eps)/deltaGnu;
            RE_IncidentSpectrum[n+group]     = (RadEnergyIncident[n+group]+eps)/deltaGnu;
            RE_EscapeRateSpectrum[n+group]   = (RadEnergyEscRate[n+group]+eps)/deltaGnu;
            RE_IncidentRateSpectrum[n+group] = (RadEnergyIncRate[n+group]+eps)/deltaGnu;
        }
    }

}



