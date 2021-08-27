#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

void Get_2d_communicator(MPI_Comm * comm2d)
{

  int i, node, world_rank, world_size, ranks_per_node, local_rank;
  int rank2d, color, key, match, err = 0;
  int Xcoord, Ycoord;                  // 2d coords in units of nodes
  int Xnodes, Ynodes;                  // 2d sizes in units of nodes
  int Bx, By;                          // 2d local box dimensions
  int Px, Py;                          // 2d logical process coordinates
  int Lx, Ly;                          // 2d local coordinates witin each node
  char label[32], host[160], *ptr, * snames, * rnames;
  FILE * fp;
  MPI_Comm local_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  snames = (char *) malloc(world_size*sizeof(host));
  rnames = (char *) malloc(world_size*sizeof(host));
  gethostname(host, sizeof(host));

  for (i=0; i<sizeof(host); i++) {    
     if (host[i] == '.') {
         host[i] = '\0';
         break;
     }   
  }

  for (i=0; i<world_size; i++)  {
    ptr = snames + i*sizeof(host);
    strncpy(ptr, host, sizeof(host));
  }

  MPI_Alltoall(snames, sizeof(host), MPI_BYTE,
               rnames, sizeof(host), MPI_BYTE, MPI_COMM_WORLD);
  color = 0;
  match = 0;
  for (i=0; i<world_size; i++) {
    ptr = rnames + i*sizeof(host);
    if (strcmp(host, ptr) == 0) {    
      match++;
      if (match == 1) color = i;
    }   
  }

  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &ranks_per_node);

  if (world_rank == 0) {
     fp = fopen("comm2d.in", "r");
     if (fp == NULL) {
        Bx = (int) sqrt((double) ranks_per_node + 0.1);
        while ( (ranks_per_node % Bx != 0) && (Bx > 0) ) Bx--;

        By = ranks_per_node / Bx;
  
        Ynodes = (int) sqrt( 0.1 + ((double) world_size) / ((double) ranks_per_node) );
        while ( (world_size % (Ynodes*ranks_per_node) != 0) && (Ynodes > 0) ) Ynodes--; 
        
        Xnodes = world_size / (Ynodes*ranks_per_node);
        
        if (world_size != (Xnodes*Ynodes*Bx*By)) {
          fprintf(stderr, "unable to construct a 2d layout ... exiting\n");
          MPI_Abort(MPI_COMM_WORLD, err);
        }
     }
     else {
        fscanf(fp, "%s %d", label, &Xnodes);
        fscanf(fp, "%s %d", label, &Ynodes);
        fscanf(fp, "%s %d", label, &Bx);
        fscanf(fp, "%s %d", label, &By);
        if (world_size != (Xnodes*Ynodes*Bx*By)) {
          fprintf(stderr, "fix the 2d layout in comm2d.in ... exiting\n");
          MPI_Abort(MPI_COMM_WORLD, err);
        }
     }
  }
  MPI_Bcast(&Xnodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ynodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Bx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&By, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // assume that nodes increment in xy order
  node = world_rank / ranks_per_node;
  Ycoord = node / Xnodes;
  Xcoord = node - Ycoord*Xnodes;

  // use xy order for local and global cartesian ranks
  // local_rank = Lx + Ly*Bx
  Ly =  local_rank / Bx;
  Lx = local_rank - Ly*Bx;

  Px = Lx + (Bx*Xcoord);
  Py = Ly + (By*Ycoord);

  key = Px + Py*(Bx*Xnodes);

  color = 1;

  MPI_Comm_split(MPI_COMM_WORLD, color, key, comm2d);

  if (world_rank == 0) fprintf(stderr, "constructed a 2d communicator with logical dimensions <%d,%d>\n", Bx*Xnodes, By*Ynodes);
/*
  MPI_Comm_rank(*comm2d, &rank2d);
  printf("rank2d %d has logical 2d coords <%d,%d>; world_rank = %d, node = %d \n", rank2d, Px, Py, world_rank, node);
*/
  return;
}


// Fortran interface
#pragma weak get_2d_communicator_=get_2d_communicator
void get_2d_communicator(int * fortran_comm2d)
{

  int i, node, world_rank, world_size, ranks_per_node, local_rank;
  int rank2d, color, key, match, err = 0;
  int Xcoord, Ycoord;                  // 2d coords in units of nodes
  int Xnodes, Ynodes;                  // 2d sizes in units of nodes
  int Bx, By;                          // 2d local box dimensions
  int Px, Py;                          // 2d logical process coordinates
  int Lx, Ly;                          // 2d local coordinates witin each node
  char label[32], host[160], *ptr, * snames, * rnames;
  FILE * fp;
  MPI_Comm comm2d, local_comm;

  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  snames = (char *) malloc(world_size*sizeof(host));
  rnames = (char *) malloc(world_size*sizeof(host));
  gethostname(host, sizeof(host));

  for (i=0; i<sizeof(host); i++) {    
     if (host[i] == '.') {
         host[i] = '\0';
         break;
     }   
  }

  for (i=0; i<world_size; i++)  {
    ptr = snames + i*sizeof(host);
    strncpy(ptr, host, sizeof(host));
  }

  MPI_Alltoall(snames, sizeof(host), MPI_BYTE,
               rnames, sizeof(host), MPI_BYTE, MPI_COMM_WORLD);
  color = 0;
  match = 0;
  for (i=0; i<world_size; i++) {
    ptr = rnames + i*sizeof(host);
    if (strcmp(host, ptr) == 0) {    
      match++;
      if (match == 1) color = i;
    }   
  }

  MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &ranks_per_node);

  if (world_rank == 0) {
     fp = fopen("comm2d.in", "r");
     if (fp == NULL) {
        Bx = (int) sqrt((double) ranks_per_node + 0.1);
        while ( (ranks_per_node % Bx != 0) && (Bx > 0) ) Bx--;

        By = ranks_per_node / Bx;
  
        Ynodes = (int) sqrt( 0.1 + ((double) world_size) / ((double) ranks_per_node) );
        while ( (world_size % (Ynodes*ranks_per_node) != 0) && (Ynodes > 0) ) Ynodes--; 

        Xnodes = world_size / (Ynodes*ranks_per_node);
        
        if (world_size != (Xnodes*Ynodes*Bx*By)) {
          fprintf(stderr, "unable to construct a 2d layout ... exiting\n");
          MPI_Abort(MPI_COMM_WORLD, err);
        }
     }
     else {
        fscanf(fp, "%s %d", label, &Xnodes);
        fscanf(fp, "%s %d", label, &Ynodes);
        fscanf(fp, "%s %d", label, &Bx);
        fscanf(fp, "%s %d", label, &By);
        if (world_size != (Xnodes*Ynodes*Bx*By)) {
          fprintf(stderr, "fix the 2d layout in comm2d.in ... exiting\n");
          MPI_Abort(MPI_COMM_WORLD, err);
        }
     }
  }
  MPI_Bcast(&Xnodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Ynodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&Bx, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&By, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // assume that nodes increment in xy order
  node = world_rank / ranks_per_node;
  Ycoord = node / Xnodes;
  Xcoord = node - Ycoord*Xnodes;

  // use xy order for local and global cartesian ranks
  // local_rank = Lx + Ly*Bx
  Ly =  local_rank / Bx;
  Lx = local_rank - Ly*Bx;

  Px = Lx + (Bx*Xcoord);
  Py = Ly + (By*Ycoord);

  key = Px + Py*(Bx*Xnodes);

  color = 1;

  MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm2d);

  *fortran_comm2d = MPI_Comm_c2f(comm2d);

  if (world_rank == 0) fprintf(stderr, "constructed a 2d communicator with logical dimensions <%d,%d>\n", Bx*Xnodes, By*Ynodes);
/*
  MPI_Comm_rank(comm2d, &rank2d);
  printf("rank2d %d has logical 2d coords <%d,%d>; world_rank = %d, node = %d \n", rank2d, Px, Py, world_rank, node);
*/
  return;
}
