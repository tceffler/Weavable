make -C CMG_CLEAN
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/CMG_CLEAN'
make -C src
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/CMG_CLEAN/src'
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o meshAndInputData.o meshAndInputData.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGDomainQuery.o CMGDomainQuery.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGIO.o CMGIO.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGGenerator.o CMGGenerator.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGMeshQuery.o CMGMeshQuery.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGGlobalMeshQuery.o CMGGlobalMeshQuery.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGTagQuery.o CMGTagQuery.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o CMGMeshTopology.o CMGMeshTopology.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o dataTypes.o dataTypes.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o fortranUtilities.o fortranUtilities.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o subdivision.o subdivision.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o lex.yy.o lex.yy.c
PGC-W-0136-Function yyerror has non-prototype declaration in scope (cmgparse.lex: 51)
PGC-W-0136-Function newLine has non-prototype declaration in scope (cmgparse.lex: 57)
PGC/power Linux 18.7-0: compilation completed with warnings
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o cmgparse.tab.o cmgparse.tab.c
ar rv libcmgp.a meshAndInputData.o CMGDomainQuery.o CMGIO.o CMGGenerator.o CMGMeshQuery.o CMGGlobalMeshQuery.o CMGTagQuery.o CMGMeshTopology.o dataTypes.o fortranUtilities.o subdivision.o lex.yy.o cmgparse.tab.o
ar: creating libcmgp.a
a - meshAndInputData.o
a - CMGDomainQuery.o
a - CMGIO.o
a - CMGGenerator.o
a - CMGMeshQuery.o
a - CMGGlobalMeshQuery.o
a - CMGTagQuery.o
a - CMGMeshTopology.o
a - dataTypes.o
a - fortranUtilities.o
a - subdivision.o
a - lex.yy.o
a - cmgparse.tab.o
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/CMG_CLEAN/src'
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/CMG_CLEAN'
make -C cmg2Kull
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/cmg2Kull'
make -C sources
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/cmg2Kull/sources'
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-CMG.o C2K-CMG.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_Geom.o C2K-KC_Geom.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_API.o C2K-KC_API.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-Lists.o C2K-Lists.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-Storage.o C2K-Storage.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_Alter.o C2K-KC_Alter.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_Create.o C2K-KC_Create.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_SubDivide.o C2K-KC_SubDivide.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_Check.o C2K-KC_Check.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_CutZone.o C2K-KC_CutZone.c
mpicc   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../../CMG_CLEAN/src -I. -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX    -c -o C2K-KC_Info.o C2K-KC_Info.c
ar rv libc2k.a C2K-CMG.o C2K-KC_Geom.o C2K-KC_API.o C2K-Lists.o C2K-Storage.o C2K-KC_Alter.o C2K-KC_Create.o C2K-KC_SubDivide.o C2K-KC_Check.o C2K-KC_CutZone.o C2K-KC_Info.o
ar: creating libc2k.a
a - C2K-CMG.o
a - C2K-KC_Geom.o
a - C2K-KC_API.o
a - C2K-Lists.o
a - C2K-Storage.o
a - C2K-KC_Alter.o
a - C2K-KC_Create.o
a - C2K-KC_SubDivide.o
a - C2K-KC_Check.o
a - C2K-KC_CutZone.o
a - C2K-KC_Info.o
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/cmg2Kull/sources'
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/cmg2Kull'
make -C Teton
make[1]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton'
make -C geom
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom'
make -C CMI
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/CMI'
g++ -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities SideBase.cc -MM -MF SideBase.d
#mpicxx -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities SideBase.cc -MM -MF SideBase.d
g++ -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities FaceBase.cc -MM -MF FaceBase.d
#mpicxx -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities FaceBase.cc -MM -MF FaceBase.d
g++ -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities CornerBase.cc -MM -MF CornerBase.d
#mpicxx -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities CornerBase.cc -MM -MF CornerBase.d
g++ -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities ZoneBase.cc -MM -MF ZoneBase.d
#mpicxx -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities ZoneBase.cc -MM -MF ZoneBase.d
g++ -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities MeshBase.cc -MM -MF MeshBase.d
#mpicxx -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities MeshBase.cc -MM -MF MeshBase.d
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/CMI'
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/CMI'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   MeshBase.cc -o MeshBase.o
"MeshBase.cc", line 41: warning: conversion from a string literal to "char *"
          is deprecated
      CheckAllConnectivityInfo ("After creating KC mesh.\n");
                                ^

"MeshBase.cc", line 46: warning: conversion from a string literal to "char *"
          is deprecated
      CheckAllConnectivityInfo ("After centroid and volume calcs.\n");
                                ^

mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   ZoneBase.cc -o ZoneBase.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   CornerBase.cc -o CornerBase.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   FaceBase.cc -o FaceBase.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   SideBase.cc -o SideBase.o
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/CMI'
make -C Field
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/Field'
g++ -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src FieldInst.cc -MM -MF FieldInst.d
#mpicxx -c   -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src FieldInst.cc -MM -MF FieldInst.d
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/Field'
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/Field'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   FieldInst.cc -o FieldInst.o
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/Field'
make -C Region
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/Region'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   RegionInst.cc -o RegionInst.o
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom/Region'
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/geom'
make -C communication
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/communication'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   DomainNeighborMapInst.cc -o DomainNeighborMapInst.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   CommAgent.cc -o CommAgent.o
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/communication'
make -C part
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/part'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   OpacityBase.cc -o OpacityBase.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   PartInst.cc -o PartInst.o
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/part'
make -C transport
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/transport'
make -C Teton
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/transport/Teton'
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/kind_mod.F90 -o mods/kind_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/constant_mod.F90 -o mods/constant_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Size_mod.F90 -o mods/Size_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c misc/mpi_param_mod.F90 -o misc/mpi_param_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c misc/mpif90_mod.F90 -o misc/mpif90_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Communicator_mod.F90 -o mods/Communicator_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Quadrature_mod.F90 -o mods/Quadrature_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/QuadratureList_mod.F90 -o mods/QuadratureList_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Boundary_mod.F90 -o mods/Boundary_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/BoundaryList_mod.F90 -o mods/BoundaryList_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Editor_mod.F90 -o mods/Editor_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/ZoneData_mod.F90 -o mods/ZoneData_mod.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'zonedata_mod_zonedata_gpu_init_mesh_kernel_' for 'sm_70'
ptxas info    : Function properties for zonedata_mod_zonedata_gpu_init_mesh_kernel_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'zonedata_mod_zonedata_gpu_update_stotal_kernel_' for 'sm_70'
ptxas info    : Function properties for zonedata_mod_zonedata_gpu_update_stotal_kernel_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 38 registers, 392 bytes cmem[0]
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Geometry_mod.F90 -o mods/Geometry_mod.o
PGF90-W-0164-Overlapping data initializations of ._dtInit4102 (mods/Geometry_mod.F90)
  0 inform,   1 warnings,   0 severes, 0 fatal for geometry_mod
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Material_mod.F90 -o mods/Material_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/Profile_mod.F90 -o mods/Profile_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/ProfileList_mod.F90 -o mods/ProfileList_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/QuadratureData_mod.F90 -o mods/QuadratureData_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/TimeStepControls_mod.F90 -o mods/TimeStepControls_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/io_mod.F90 -o mods/io_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/iter_control_mod.F90 -o mods/iter_control_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/iter_control_list_mod.F90 -o mods/iter_control_list_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/radconstant_mod.F90 -o mods/radconstant_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/nvtx_mod.F90 -o mods/nvtx_mod.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c mods/GPUhelper_mod.F90 -o mods/GPUhelper_mod.o
ptxas info    : 432 bytes gmem
ptxas info    : Compiling entry function 'gpuhelper_mod_computestimed_' for 'sm_70'
ptxas info    : Function properties for gpuhelper_mod_computestimed_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 30 registers, 392 bytes cmem[0]
ptxas info    : Compiling entry function 'computestime_1121_gpu' for 'sm_70'
ptxas info    : Function properties for computestime_1121_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 27 registers, 416 bytes cmem[0]
ptxas info    : Compiling entry function 'scalepsibyvolume_1071_gpu' for 'sm_70'
ptxas info    : Function properties for scalepsibyvolume_1071_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 34 registers, 408 bytes cmem[0]
ptxas info    : Compiling entry function 'gpuhelper_mod_gpu_fp_ez_hplane_f_' for 'sm_70'
ptxas info    : Function properties for gpuhelper_mod_gpu_fp_ez_hplane_f_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 64 registers, 448 bytes cmem[0]
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c misc/assert.F90 -o misc/assert.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c misc/f90advise.F90 -o misc/f90advise.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c misc/f90fatal.F90 -o misc/f90fatal.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructBoundary.F90 -o aux/ConstructBoundary.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructDtControls.F90 -o aux/ConstructDtControls.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructEditor.F90 -o aux/ConstructEditor.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructGeometry.F90 -o aux/ConstructGeometry.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructIterControls.F90 -o aux/ConstructIterControls.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructMaterial.F90 -o aux/ConstructMaterial.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructProfile.F90 -o aux/ConstructProfile.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructQuadrature.F90 -o aux/ConstructQuadrature.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ConstructSize.F90 -o aux/ConstructSize.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/ResetSize.F90 -o aux/ResetSize.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/addBoundary.F90 -o aux/addBoundary.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/addProfile.F90 -o aux/addProfile.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/getEdits.F90 -o aux/getEdits.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/getGeometry.F90 -o aux/getGeometry.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/getRunStats.F90 -o aux/getRunStats.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setEditorModule.F90 -o aux/setEditorModule.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setEnergyEdits.F90 -o aux/setEnergyEdits.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setGeometry.F90 -o aux/setGeometry.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setMaterialModule.F90 -o aux/setMaterialModule.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setSnOrder.F90 -o aux/setSnOrder.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setTimeStep.F90 -o aux/setTimeStep.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c aux/setZone.F90 -o aux/setZone.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/RadMoments.F90 -o control/RadMoments.o
pgf90 -Mcuda=cc70,nordc,maxregcount:80,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical  -c snac/snswp3d.F90 -o snac/snswp3d.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'setexitfluxd2_461_gpu' for 'sm_70'
ptxas info    : Function properties for setexitfluxd2_461_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 49 registers, 440 bytes cmem[0]
ptxas info    : Compiling entry function 'snswp3d_mod_setexitfluxd_' for 'sm_70'
ptxas info    : Function properties for snswp3d_mod_setexitfluxd_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 57 registers, 416 bytes cmem[0]
ptxas info    : Compiling entry function 'snswp3d_mod_gpu_sweep_' for 'sm_70'
ptxas info    : Function properties for snswp3d_mod_gpu_sweep_
    208 bytes stack frame, 52 bytes spill stores, 72 bytes spill loads
ptxas info    : Used 80 registers, 472 bytes cmem[0], 24 bytes cmem[2]
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/advanceRT.F90 -o control/advanceRT.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/dtnew.F90 -o control/dtnew.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/newenergy.F90 -o control/newenergy.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/profint.F90 -o control/profint.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/radtr.F90 -o control/radtr.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/rtbatch.F90 -o control/rtbatch.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/rtbdry.F90 -o control/rtbdry.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/rtedit.F90 -o control/rtedit.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/rtinit.F90 -o control/rtinit.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c control/rtvsrc.F90 -o control/rtvsrc.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/InitExchange.F90 -o rt/InitExchange.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/SweepScheduler.F90 -o rt/SweepScheduler.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/UpdateMaterialCoupling.F90 -o rt/UpdateMaterialCoupling.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/bdyedt.F90 -o rt/bdyedt.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/exchange.F90 -o rt/exchange.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/face_coords.F90 -o rt/face_coords.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/findReflectedAngles.F90 -o rt/findReflectedAngles.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/findexit.F90 -o rt/findexit.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/getAbsorptionRate.F90 -o rt/getAbsorptionRate.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/initcomm.F90 -o rt/initcomm.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/quadLobatto.F90 -o rt/quadLobatto.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/quadProduct.F90 -o rt/quadProduct.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/quadxyz.F90 -o rt/quadxyz.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rswpmd.F90 -o rt/rswpmd.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtave.F90 -o rt/rtave.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtcompton.F90 -o rt/rtcompton.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtconi.F90 -o rt/rtconi.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtconv.F90 -o rt/rtconv.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtgeom3.F90 -o rt/rtgeom3.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtmainsn.F90 -o rt/rtmainsn.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtorder.F90 -o rt/rtorder.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtplnk.F90 -o rt/rtplnk.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtquad.F90 -o rt/rtquad.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/rtstrtsn.F90 -o rt/rtstrtsn.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/setIncidentFlux.F90 -o rt/setIncidentFlux.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/setbdy.F90 -o rt/setbdy.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'setbdyd_132_gpu' for 'sm_70'
ptxas info    : Function properties for setbdyd_132_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 37 registers, 460 bytes cmem[0]
ptxas info    : Compiling entry function 'setbdyd_97_gpu' for 'sm_70'
ptxas info    : Function properties for setbdyd_97_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 21 registers, 400 bytes cmem[0]
ptxas info    : Compiling entry function 'setbdyd_78_gpu' for 'sm_70'
ptxas info    : Function properties for setbdyd_78_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 37 registers, 460 bytes cmem[0]
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/testFluxConv.F90 -o rt/testFluxConv.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c rt/zone_coords.F90 -o rt/zone_coords.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/GaussLegendreLobattoWgts.F90 -o snac/GaussLegendreLobattoWgts.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/Jacobi.F90 -o snac/Jacobi.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/JacobiLobattoPts.F90 -o snac/JacobiLobattoPts.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/cyclebreaker.F90 -o snac/cyclebreaker.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/findseeds.F90 -o snac/findseeds.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/fixZone.F90 -o snac/fixZone.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/getDownStreamData.F90 -o snac/getDownStreamData.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/sccsearch.F90 -o snac/sccsearch.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snmoments.F90 -o snac/snmoments.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'snmomentsd_127_gpu' for 'sm_70'
ptxas info    : Function properties for snmomentsd_127_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 61 registers, 440 bytes cmem[0]
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snmref.F90 -o snac/snmref.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snneed.F90 -o snac/snneed.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snnext.F90 -o snac/snnext.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snpnmset.F90 -o snac/snpnmset.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snreflect.F90 -o snac/snreflect.o
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'snreflectd_87_gpu' for 'sm_70'
ptxas info    : Function properties for snreflectd_87_gpu
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 29 registers, 448 bytes cmem[0]
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snflwxyz.F90 -o snac/snflwxyz.o
pgf90 -DLINUX,-DLinux,-Dmpi,-DMPI,-DUSE_MPI,-DOMPI_SKIP_MPICXX  -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I./include -Mcuda=cc70,nordc,maxregcount:64,ptxinfo,loadcache:L1,cuda9.2 -fast -Mfprelaxed -O3 -Kpic -mp -Munixlogical   -c snac/snynmset.F90 -o snac/snynmset.o
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/transport/Teton'
make -C TetonInterface
make[3]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/transport/TetonInterface'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   TetonInst.cc -o TetonInst.o
"../../transport/TetonInterface/Teton.cc", line 468: warning: variable
          "stringLen" was declared but never referenced
      long int stringLen=8;
               ^

"../../transport/TetonInterface/Teton.cc", line 758: warning: variable "c2" was
          declared but never referenced
      int Onode1 = -1, c1 = -1, c2 = -1, Ofid = -1;
                                ^

"../../transport/TetonInterface/Teton.cc", line 760: warning: variable
          "Ocorner2" was declared but never referenced
      int Ocorner2 = -1, Onode2 = -1, node2 = -1;
          ^

"../../transport/TetonInterface/Teton.cc", line 760: warning: variable "Onode2"
          was declared but never referenced
      int Ocorner2 = -1, Onode2 = -1, node2 = -1;
                         ^

"../../transport/TetonInterface/Teton.cc", line 760: warning: variable "node2"
          was declared but never referenced
      int Ocorner2 = -1, Onode2 = -1, node2 = -1;
                                      ^

"../../transport/TetonInterface/Teton.cc", line 1685: warning: variable "oldSE"
          was declared but never referenced
     double deltaSE, totalME, delME, oldSE, frac, VFmult;
                                     ^

"../../transport/TetonInterface/Teton.cc", line 1685: warning: variable "frac"
          was declared but never referenced
     double deltaSE, totalME, delME, oldSE, frac, VFmult;
                                            ^

mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I../.. -I../../../cmg2Kull/sources -I../../../CMG_CLEAN/src -I../../utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   TetonNT.cc -o TetonNT.o
make[3]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/transport/TetonInterface'
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/transport'
make -C utilities
make[2]: Entering directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/utilities'
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   DBC.cc -o DBC.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   VERIFY.cc -o VERIFY.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I.. -I../../cmg2Kull/sources -I../../CMG_CLEAN/src -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   Process.cc -o Process.o
make[2]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/utilities'
ar rv libInfrastructure.a communication/DomainNeighborMapInst.o communication/CommAgent.o geom/Region/RegionInst.o geom/Field/FieldInst.o geom/CMI/MeshBase.o geom/CMI/ZoneBase.o geom/CMI/CornerBase.o geom/CMI/FaceBase.o geom/CMI/SideBase.o part/OpacityBase.o part/PartInst.o transport/TetonInterface/TetonNT.o transport/TetonInterface/TetonInst.o transport/Teton/mods/BoundaryList_mod.o transport/Teton/mods/Boundary_mod.o transport/Teton/mods/Communicator_mod.o transport/Teton/mods/Editor_mod.o transport/Teton/mods/Geometry_mod.o transport/Teton/mods/Material_mod.o transport/Teton/mods/ProfileList_mod.o transport/Teton/mods/Profile_mod.o transport/Teton/mods/QuadratureData_mod.o transport/Teton/mods/QuadratureList_mod.o transport/Teton/mods/Quadrature_mod.o transport/Teton/mods/Size_mod.o transport/Teton/mods/TimeStepControls_mod.o transport/Teton/mods/ZoneData_mod.o transport/Teton/mods/constant_mod.o transport/Teton/mods/io_mod.o transport/Teton/mods/iter_control_list_mod.o transport/Teton/mods/iter_control_mod.o transport/Teton/mods/kind_mod.o transport/Teton/mods/radconstant_mod.o transport/Teton/mods/nvtx_mod.o transport/Teton/mods/GPUhelper_mod.o transport/Teton/misc/assert.o transport/Teton/misc/f90advise.o transport/Teton/misc/f90fatal.o transport/Teton/misc/mpi_param_mod.o transport/Teton/misc/mpif90_mod.o transport/Teton/aux/ConstructBoundary.o transport/Teton/aux/ConstructDtControls.o transport/Teton/aux/ConstructEditor.o transport/Teton/aux/ConstructGeometry.o transport/Teton/aux/ConstructIterControls.o transport/Teton/aux/ConstructMaterial.o transport/Teton/aux/ConstructProfile.o transport/Teton/aux/ConstructQuadrature.o transport/Teton/aux/ConstructSize.o transport/Teton/aux/ResetSize.o transport/Teton/aux/addBoundary.o transport/Teton/aux/addProfile.o transport/Teton/aux/getEdits.o transport/Teton/aux/getGeometry.o transport/Teton/aux/getRunStats.o transport/Teton/aux/setEditorModule.o transport/Teton/aux/setEnergyEdits.o transport/Teton/aux/setGeometry.o transport/Teton/aux/setMaterialModule.o transport/Teton/aux/setSnOrder.o transport/Teton/aux/setTimeStep.o transport/Teton/aux/setZone.o transport/Teton/control/RadMoments.o transport/Teton/control/advanceRT.o transport/Teton/control/dtnew.o transport/Teton/control/newenergy.o transport/Teton/control/profint.o transport/Teton/control/radtr.o transport/Teton/control/rtbatch.o transport/Teton/control/rtbdry.o transport/Teton/control/rtedit.o transport/Teton/control/rtinit.o transport/Teton/control/rtvsrc.o transport/Teton/rt/InitExchange.o transport/Teton/rt/SweepScheduler.o transport/Teton/rt/UpdateMaterialCoupling.o transport/Teton/rt/bdyedt.o transport/Teton/rt/exchange.o transport/Teton/rt/face_coords.o transport/Teton/rt/findReflectedAngles.o transport/Teton/rt/findexit.o transport/Teton/rt/getAbsorptionRate.o transport/Teton/rt/initcomm.o transport/Teton/rt/quadLobatto.o transport/Teton/rt/quadProduct.o transport/Teton/rt/quadxyz.o transport/Teton/rt/rswpmd.o transport/Teton/rt/rtave.o transport/Teton/rt/rtcompton.o transport/Teton/rt/rtconi.o transport/Teton/rt/rtconv.o transport/Teton/rt/rtgeom3.o transport/Teton/rt/rtmainsn.o transport/Teton/rt/rtorder.o transport/Teton/rt/rtplnk.o transport/Teton/rt/rtquad.o transport/Teton/rt/rtstrtsn.o transport/Teton/rt/setIncidentFlux.o transport/Teton/rt/setbdy.o transport/Teton/rt/testFluxConv.o transport/Teton/rt/zone_coords.o transport/Teton/snac/GaussLegendreLobattoWgts.o transport/Teton/snac/Jacobi.o transport/Teton/snac/JacobiLobattoPts.o transport/Teton/snac/cyclebreaker.o transport/Teton/snac/findseeds.o transport/Teton/snac/fixZone.o transport/Teton/snac/getDownStreamData.o transport/Teton/snac/sccsearch.o transport/Teton/snac/snflwxyz.o transport/Teton/snac/snmoments.o transport/Teton/snac/snmref.o transport/Teton/snac/snneed.o transport/Teton/snac/snnext.o transport/Teton/snac/snpnmset.o transport/Teton/snac/snreflect.o transport/Teton/snac/snswp3d.o transport/Teton/snac/snynmset.o utilities/VERIFY.o utilities/DBC.o utilities/Process.o
ar: creating libInfrastructure.a
a - communication/DomainNeighborMapInst.o
a - communication/CommAgent.o
a - geom/Region/RegionInst.o
a - geom/Field/FieldInst.o
a - geom/CMI/MeshBase.o
a - geom/CMI/ZoneBase.o
a - geom/CMI/CornerBase.o
a - geom/CMI/FaceBase.o
a - geom/CMI/SideBase.o
a - part/OpacityBase.o
a - part/PartInst.o
a - transport/TetonInterface/TetonNT.o
a - transport/TetonInterface/TetonInst.o
a - transport/Teton/mods/BoundaryList_mod.o
a - transport/Teton/mods/Boundary_mod.o
a - transport/Teton/mods/Communicator_mod.o
a - transport/Teton/mods/Editor_mod.o
a - transport/Teton/mods/Geometry_mod.o
a - transport/Teton/mods/Material_mod.o
a - transport/Teton/mods/ProfileList_mod.o
a - transport/Teton/mods/Profile_mod.o
a - transport/Teton/mods/QuadratureData_mod.o
a - transport/Teton/mods/QuadratureList_mod.o
a - transport/Teton/mods/Quadrature_mod.o
a - transport/Teton/mods/Size_mod.o
a - transport/Teton/mods/TimeStepControls_mod.o
a - transport/Teton/mods/ZoneData_mod.o
a - transport/Teton/mods/constant_mod.o
a - transport/Teton/mods/io_mod.o
a - transport/Teton/mods/iter_control_list_mod.o
a - transport/Teton/mods/iter_control_mod.o
a - transport/Teton/mods/kind_mod.o
a - transport/Teton/mods/radconstant_mod.o
a - transport/Teton/mods/nvtx_mod.o
a - transport/Teton/mods/GPUhelper_mod.o
a - transport/Teton/misc/assert.o
a - transport/Teton/misc/f90advise.o
a - transport/Teton/misc/f90fatal.o
a - transport/Teton/misc/mpi_param_mod.o
a - transport/Teton/misc/mpif90_mod.o
a - transport/Teton/aux/ConstructBoundary.o
a - transport/Teton/aux/ConstructDtControls.o
a - transport/Teton/aux/ConstructEditor.o
a - transport/Teton/aux/ConstructGeometry.o
a - transport/Teton/aux/ConstructIterControls.o
a - transport/Teton/aux/ConstructMaterial.o
a - transport/Teton/aux/ConstructProfile.o
a - transport/Teton/aux/ConstructQuadrature.o
a - transport/Teton/aux/ConstructSize.o
a - transport/Teton/aux/ResetSize.o
a - transport/Teton/aux/addBoundary.o
a - transport/Teton/aux/addProfile.o
a - transport/Teton/aux/getEdits.o
a - transport/Teton/aux/getGeometry.o
a - transport/Teton/aux/getRunStats.o
a - transport/Teton/aux/setEditorModule.o
a - transport/Teton/aux/setEnergyEdits.o
a - transport/Teton/aux/setGeometry.o
a - transport/Teton/aux/setMaterialModule.o
a - transport/Teton/aux/setSnOrder.o
a - transport/Teton/aux/setTimeStep.o
a - transport/Teton/aux/setZone.o
a - transport/Teton/control/RadMoments.o
a - transport/Teton/control/advanceRT.o
a - transport/Teton/control/dtnew.o
a - transport/Teton/control/newenergy.o
a - transport/Teton/control/profint.o
a - transport/Teton/control/radtr.o
a - transport/Teton/control/rtbatch.o
a - transport/Teton/control/rtbdry.o
a - transport/Teton/control/rtedit.o
a - transport/Teton/control/rtinit.o
a - transport/Teton/control/rtvsrc.o
a - transport/Teton/rt/InitExchange.o
a - transport/Teton/rt/SweepScheduler.o
a - transport/Teton/rt/UpdateMaterialCoupling.o
a - transport/Teton/rt/bdyedt.o
a - transport/Teton/rt/exchange.o
a - transport/Teton/rt/face_coords.o
a - transport/Teton/rt/findReflectedAngles.o
a - transport/Teton/rt/findexit.o
a - transport/Teton/rt/getAbsorptionRate.o
a - transport/Teton/rt/initcomm.o
a - transport/Teton/rt/quadLobatto.o
a - transport/Teton/rt/quadProduct.o
a - transport/Teton/rt/quadxyz.o
a - transport/Teton/rt/rswpmd.o
a - transport/Teton/rt/rtave.o
a - transport/Teton/rt/rtcompton.o
a - transport/Teton/rt/rtconi.o
a - transport/Teton/rt/rtconv.o
a - transport/Teton/rt/rtgeom3.o
a - transport/Teton/rt/rtmainsn.o
a - transport/Teton/rt/rtorder.o
a - transport/Teton/rt/rtplnk.o
a - transport/Teton/rt/rtquad.o
a - transport/Teton/rt/rtstrtsn.o
a - transport/Teton/rt/setIncidentFlux.o
a - transport/Teton/rt/setbdy.o
a - transport/Teton/rt/testFluxConv.o
a - transport/Teton/rt/zone_coords.o
a - transport/Teton/snac/GaussLegendreLobattoWgts.o
a - transport/Teton/snac/Jacobi.o
a - transport/Teton/snac/JacobiLobattoPts.o
a - transport/Teton/snac/cyclebreaker.o
a - transport/Teton/snac/findseeds.o
a - transport/Teton/snac/fixZone.o
a - transport/Teton/snac/getDownStreamData.o
a - transport/Teton/snac/sccsearch.o
a - transport/Teton/snac/snflwxyz.o
a - transport/Teton/snac/snmoments.o
a - transport/Teton/snac/snmref.o
a - transport/Teton/snac/snneed.o
a - transport/Teton/snac/snnext.o
a - transport/Teton/snac/snpnmset.o
a - transport/Teton/snac/snreflect.o
a - transport/Teton/snac/snswp3d.o
a - transport/Teton/snac/snynmset.o
a - utilities/VERIFY.o
a - utilities/DBC.o
a - utilities/Process.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../cmg2Kull/sources -I../CMG_CLEAN/src -I/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   TetonUtils.cc -o TetonUtils.o
"TetonUtils.cc", line 51: warning: variable "myRank" was declared but never
          referenced
      int myRank=0,theNumOmpThreads=-1,myThreadNum=-2;
          ^

"TetonUtils.cc", line 51: warning: variable "theNumOmpThreads" was declared but
          never referenced
      int myRank=0,theNumOmpThreads=-1,myThreadNum=-2;
                   ^

"TetonUtils.cc", line 51: warning: variable "myThreadNum" was declared but
          never referenced
      int myRank=0,theNumOmpThreads=-1,myThreadNum=-2;
                                       ^

"TetonUtils.cc", line 203: warning: variable "timeRadtr" was declared but never
          referenced
      double timeRadtr=0.0,radtotal=0.0;
             ^

"TetonUtils.cc", line 203: warning: variable "radtotal" was declared but never
          referenced
      double timeRadtr=0.0,radtotal=0.0;
                           ^

"TetonUtils.cc", line 261: warning: variable "a" was declared but never
          referenced
      double  a = alpha_cv/4.0;    
              ^

ar rv libTetonUtils.a TetonUtils.o
ar: creating libTetonUtils.a
a - TetonUtils.o
mpicxx -c   -I//autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/pgi-18.7/spectrum-mpi-10.3.1.2-20200121-kvospi2rdlh4mxnlsrfatkmtnmgl63wi/include -I/sw/summit/cuda/9.1.85/include -I. -I../cmg2Kull/sources -I../CMG_CLEAN/src -I/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton/utilities -g -O2 -fpic -mp  -DLINUX -DLinux -DUSE_MPI -DOMPI_SKIP_MPICXX   TetonTest.cc -o TetonTest.o
make[1]: Leaving directory `/autofs/nccs-svm1_proj/csc425/olcf-acceptance-harness/applications/new-harness/summit/CORAL-UMT/Source/umt2016-managed-pgi18.7/Teton'
