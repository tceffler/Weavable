#include <arpa/inet.h>
#include "mpi.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <infiniband/verbs.h>
#include <sys/time.h>

#define RDMA_WRID 3

// if x is NON-ZERO, error is printed
#define TEST_NZ(x,y) do { if ((x)) die(y); } while (0)

// if x is ZERO, error is printed
#define TEST_Z(x,y) do { if (!(x)) die(y); } while (0)

// if x is NEGATIVE, error is printed
#define TEST_N(x,y) do { if ((x)<0) die(y); } while (0)

static int die(const char *reason);
static void qp_change_state_rts();
static void qp_change_state_rtr();
void query_qp();

struct ib_connection {
    int             	lid;
    int*       	 	qpnV;
    int       	 	qpn;
    int*             	psnV;
    int             	psn;
    unsigned 		rkey;
    unsigned long long    	vaddr;
};

struct app_context{
	struct ibv_context 		*context;
	struct ibv_pd      		*pd;
	struct ibv_mr      		*mr;
        int ncq;
	struct ibv_cq      		**cq;
        int nQPair;
	struct ibv_qp      		**qp;
	//struct ibv_comp_channel *ch;
	void               		*buf;
	size_t            	bufSize;
	//int                 	tx_depth;
	struct ibv_sge      	sge_list;
	struct ibv_send_wr  	wr;

	struct ibv_send_wr  	*bad_sr;
	struct ibv_recv_wr  	*bad_rr;

	int ib_port;
	struct ib_connection		lConn;
	struct ib_connection 		*rConn;
	struct ibv_device		*ib_dev;
        int *nMsgQ;
};

typedef struct app_context app_context;

static app_context *ctx=NULL;

//post n dummy recv in Q
void post_recv(int n,int Q)
{
  struct ibv_recv_wr wr;
  memset(&wr,0,sizeof(wr));
  
  wr.wr_id = Q;
  wr.sg_list = NULL;
  wr.num_sge = 0;
  
  for(int ii=0;ii<n;ii++) 
     TEST_NZ(ibv_post_recv(ctx->qp[Q],&wr,&ctx->bad_rr),"pose recv fail");
  //printf("Q=%d recv %d at qpn = %d \n",Q,n,ctx->qp[Q]->qp_num);
}

void post_send(unsigned int imm, int Q, size_t Soffset, size_t Roffset, size_t size)
{
  //according to RDMAmojo, work request can be reused
  //that is the following info is copied into the queue
  //
    //int sd = sizeof(double);
    size_t sd = 1;
    //printf("send offset=%zu %zu size=%zu\n",Soffset,Roffset,size);

    ctx->sge_list.addr      = (uintptr_t)(ctx->buf + sd*Soffset);
    ctx->sge_list.length    = sd*size;
    ctx->sge_list.lkey      = ctx->mr->lkey;

    ctx->wr.wr.rdma.remote_addr = (ctx->rConn[Q].vaddr + sd*Roffset);
    ctx->wr.wr.rdma.rkey        = ctx->rConn[Q].rkey;
    ctx->wr.wr_id       = Q;
    ctx->wr.sg_list     = &ctx->sge_list;
    ctx->wr.num_sge     = 1;
    ctx->wr.opcode      = IBV_WR_RDMA_WRITE_WITH_IMM;
    ctx->wr.imm_data = imm; // seq . block ( 4bit ) . dir ( 1 bit ) . chunk ( 3 bit ) 
    ctx->wr.send_flags  = IBV_SEND_SIGNALED;
    ctx->wr.next        = NULL;

    ctx->nMsgQ[Q]++;
    //printf("post in Q %d and %d to qid=%d from %x to %x\n",Q,ctx->nMsgQ[Q],ctx->rConn[Q].qpn,ctx->sge_list.addr,ctx->wr.wr.rdma.remote_addr);
    TEST_NZ(ibv_post_send(ctx->qp[Q],&ctx->wr,&ctx->bad_sr),
        "ibv_post_send failed. This is bad mkay");
  //printf("send to qpn = %d\n",ctx->qp[Q]->qp_num);
}

int check_recv(unsigned int *recvV)
{
  int rcnt=0;
  struct ibv_wc wc[10];
  for(int ii=0;ii<4;ii++)
  {
     int nrecv = ibv_poll_cq(ctx->cq[ii],10,wc);
     //printf("num of recv=%d\n",nrecv);
     if(nrecv <0) printf("poll cq failed\n");
     for(int jj=0;jj<nrecv;jj++)
     {
       if ( wc[jj].status == IBV_WC_SUCCESS )
       {
         recvV[rcnt] = wc[jj].imm_data;
         //printf("recved %x\n",wc[jj].imm_data);
       }
       else
       {
          fprintf(stdout, "Failed status %s (%d) for wr_id %d\n", ibv_wc_status_str(wc[jj].status), wc[jj].status, (int)wc[jj].wr_id);
       }
       rcnt++;
     }
  }
  return rcnt;
}

int post_control(int Qu)
{
  struct ibv_wc wc[10];
  int nCompl = ibv_poll_cq(ctx->cq[Qu%4+4],10,wc);
  for(int jj=0;jj<nCompl;jj++) 
  {
     if ( wc[jj].status == IBV_WC_SUCCESS )
     {
       int Q=wc[jj].wr_id;
       ctx->nMsgQ[Q]--;
       //printf("message at Q %d is send. %d more to send\n",Q,ctx->nMsgQ[Q]);
     }
     else
     {
        printf("dump status\n");
        printf(" wr_id=%lu    status=%u    opcode=%u    vendor_err=%u    byte_len=%u    imm_data=%u    qp_num=%u    src_qp=%u    wc_flags=%d    pkey_index=%u    slid=%u    sl=%u    dlid_path_bits=%u  \n", wc[jj].wr_id, wc[jj].status, wc[jj].opcode, wc[jj].vendor_err, wc[jj].byte_len, wc[jj].imm_data, wc[jj].qp_num, wc[jj].src_qp, wc[jj].wc_flags, wc[jj].pkey_index, wc[jj].slid, wc[jj].sl, wc[jj].dlid_path_bits);
        die("fail to send");
     }
  }
  if(ctx->nMsgQ[Qu]>60) printf("at Q, there are too many\n");
  return ctx->nMsgQ[Qu]<60?1:0;
}

void drain_send()
{
  struct ibv_wc wc[10];
  for(int ii=0;ii<4;ii++)
  {
     int nrecv = ibv_poll_cq(ctx->cq[ii+4],10,wc);
     for(int jj=0;jj<nrecv;jj++) 
     {
       if ( wc[jj].status == IBV_WC_SUCCESS )
       {
         int Q=wc[jj].wr_id;
         ctx->nMsgQ[Q]--;
         printf("message at Q %d is send. %d more to send\n",Q,ctx->nMsgQ[Q]);
       }
       else
       {
          die("fail to send");
       }
     }
  }
}

//static void destroy_ctx(struct app_context *ctx);
void destroy_ib(void);
  
void setup_ib_yz(double* h_Buf, size_t h_BufSize, int npey, int npez, int pey, int pez, int NA, int NG, int NC, int nTBG, MPI_Comm y_comm, MPI_Comm z_comm,int myrank) 
{

  int tbY=4;
  int tbZ=5;
  int pid = getpid();
  srand48(pid * time(NULL));

  //app_context *ctx;
  char        *ib_devname = NULL;
  struct ibv_qp			*qp;
  struct ibv_device *ib_dev;

  ctx = malloc(sizeof *ctx); memset(ctx, 0, sizeof *ctx);
  ctx->buf = h_Buf;
  ctx->bufSize = h_BufSize;
	
  struct ibv_device **dev_list;
  TEST_Z(dev_list = ibv_get_device_list(NULL), "No IB-device available. get_device_list returned NULL");
  TEST_Z(ctx->ib_dev = dev_list[myrank], "IB-device could not be assigned. Maybe dev_list array is empty");
  TEST_Z(ctx->context = ibv_open_device(ctx->ib_dev), "Could not create context, ibv_open_device");
  TEST_Z(ctx->pd = ibv_alloc_pd(ctx->context), "Could not allocate protection domain, ibv_alloc_pd");

  TEST_Z(ctx->mr = ibv_reg_mr(ctx->pd, ctx->buf, ctx->bufSize, IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE ),
			"Could not allocate mr, ibv_reg_mr. Do you have root access?");
	
//  printf("mr registered\n");
  //TEST_Z(ctx->ch = ibv_create_comp_channel(ctx->context), "Could not create completion channel, ibv_create_comp_channel");

  //int nMaxReq=10;
  ctx->ncq = 8;
  ctx->cq = (struct ibv_cq**)malloc(sizeof(struct ibv_cq*)*ctx->ncq);
  for(int ii=0;ii<ctx->ncq;ii++)
  {
    TEST_Z(ctx->cq[ii] = ibv_create_cq(ctx->context,1000, NULL, NULL, 0), "Could not create receive completion queue, ibv_create_cq");	
  }
//  printf("%d cq created\n",ctx->ncq);

  //creating QPair
  //QP 0,1 for y+,y-
  //QP 2 3 for z+,z-
  int nQPair = 4*nTBG;
  ctx->nQPair = nQPair;
  ctx->qp = (struct ibv_qp**)malloc(sizeof(struct ibv_qp*)*nQPair);
  ctx->nMsgQ = (int*)malloc(sizeof(int)*nQPair);
  for(int ii=0;ii<nQPair;ii++) ctx->nMsgQ[ii]=0;

  int nMaxSreq = 4 * 4 * 4;
  #define max(x,y) (x>y?x:y)
  int nMaxRreq = max(tbY,tbZ) * NC * ((int)((NA*NG)/nTBG) + 1);

  for(int ii=0;ii<nQPair;ii++)
  {
      struct ibv_qp_init_attr qp_init_attr = {
		.send_cq = ctx->cq[ii%4+4],
		.recv_cq = ctx->cq[ii%4],
		.qp_type = IBV_QPT_RC,
		.cap = {
			.max_send_wr = nMaxSreq,
			.max_recv_wr = nMaxRreq,
			.max_send_sge = 1,
			.max_recv_sge = 1,
			.max_inline_data = 0
		}
      };
      TEST_Z(ctx->qp[ii] = ibv_create_qp(ctx->pd, &qp_init_attr),
			"Could not create queue pair, ibv_create_qp");	

      ctx->ib_port = 1; //why?
      struct ibv_qp_attr attr = {
                 .qp_state        	= IBV_QPS_INIT,
                 .pkey_index      	= 0,
                 .port_num        	= ctx->ib_port,
                 .qp_access_flags	= IBV_ACCESS_REMOTE_WRITE
      };
      TEST_NZ(ibv_modify_qp(ctx->qp[ii], &attr,
                            IBV_QP_STATE        |
                            IBV_QP_PKEY_INDEX   |
                            IBV_QP_PORT         |
                            IBV_QP_ACCESS_FLAGS),
            "Could not modify QP to INIT, ibv_modify_qp");
   }
//   printf("qp done\n");

   struct ibv_port_attr port_attr;
   TEST_NZ(ibv_query_port(ctx->context,ctx->ib_port,&port_attr),
		"Could not get port attributes, ibv_query_port");

   ctx->lConn.lid = port_attr.lid;
   ctx->lConn.qpnV = (int*)malloc(sizeof(int)*nQPair); 
   for(int ii=0;ii<nQPair;ii++) ctx->lConn.qpnV[ii]=ctx->qp[ii]->qp_num;
   ctx->lConn.psnV = (int*)malloc(sizeof(int)*nQPair); 
   for(int ii=0;ii<nQPair;ii++) ctx->lConn.psnV[ii]=lrand48() & 0xffffff;
   ctx->lConn.rkey = ctx->mr->rkey;
   ctx->lConn.vaddr = (uintptr_t)ctx->buf;

   //exchange ib info
  char msg[sizeof "0000:000000:000000:00000000:0000000000000000"];
  char buf_msg[sizeof "0000:000000:000000:00000000:0000000000000000"];


  int yplus = (pey+1)%npey;
  int yminus =(pey-1+npey)%npey;

  int zplus = (pez+1)%npez;
  int zminus = (pez-1+npez)%npez;

  int dirs[] = {yplus,yminus,zplus,zminus};

//  printf("dir= %d %d %d %d \n",dirs[0],dirs[1],dirs[2],dirs[3]);

  MPI_Status Stat;
  ctx->rConn = (struct ib_connection*)malloc(sizeof(struct ib_connection)*nQPair); //4 remote QP's
  struct ib_connection *local = &ctx->lConn;
  MPI_Comm curComm;
  for(int tt=0;tt<nTBG;tt++)
  {
    for(int ii=0;ii<2;ii++)  // y/z
    {
      if (ii==0) curComm=y_comm;
      else curComm=z_comm;
 
      if( (ii==0 && npey !=1) || (ii==1 && npez !=1)) //in case there is only 1 node in the dir
      {
        int rc,parsed;
        for(int jj=0;jj<2;jj++)  //+/-
        {
          sprintf(msg, "%04x:%06x:%06x:%08x:%016Lx", local->lid, local->qpnV[tt*4+ii*2+jj], local->psnV[tt*4+ii*2+jj], local->rkey, local->vaddr);
          rc = MPI_Send(msg, sizeof(msg), MPI_CHAR, dirs[ii*2+jj], 0, curComm);
          rc = MPI_Recv(buf_msg, sizeof(msg), MPI_CHAR, dirs[ii*2+1-jj], 0, curComm, &Stat);
  
         struct ib_connection *remote = &(ctx->rConn[tt*4+ii*2+1-jj]);
         parsed = sscanf(buf_msg, "%x:%x:%x:%x:%Lx", &remote->lid, &remote->qpn, &remote->psn, &remote->rkey, &remote->vaddr);
         if(parsed != 5) fprintf(stderr, "Could not parse message from peer");
//         printf("recved from %d: %s\n",dirs[ii*2+1-jj],buf_msg);
        }
      }
    }
  }
//  printf("remote Q recved\n");
  qp_change_state_rtr();
//  printf("Q went RTR state\n");
  qp_change_state_rts();
//  printf("Q went RTS state\n");
  //query_qp();
}



static void qp_change_state_rtr()
{
    //struct ibv_qp_attr *attr;
    //attr = (struct ibv_qp_attr*) malloc( sizeof(struct ibv_qp_attr) );
    //if( attr == NULL ) die("rtr alloc fail");

    //memset(attr, 0, sizeof(struct ibv_qp_attr) );
    int sl=1;
    struct ibv_qp_attr attrPool[100];

    for(int ii=0;ii<ctx->nQPair;ii++)
    {
      struct ibv_qp_attr *attr=&attrPool[ii];
      memset(attr, 0, sizeof(struct ibv_qp_attr) );

      attr->qp_state              = IBV_QPS_RTR;
      attr->path_mtu              = IBV_MTU_2048;
      attr->dest_qp_num           = ctx->rConn[ii].qpn;
      attr->rq_psn                = ctx->rConn[ii].psn;
      attr->max_dest_rd_atomic    = 1;
      attr->min_rnr_timer         = 12;
      attr->ah_attr.is_global     = 0;
      attr->ah_attr.dlid          = ctx->rConn[ii].lid;
      attr->ah_attr.sl            = sl;
      attr->ah_attr.src_path_bits = 0;
      attr->ah_attr.port_num      = ctx->ib_port;
    
      TEST_NZ(ibv_modify_qp(ctx->qp[ii], attr,
                IBV_QP_STATE                |
                IBV_QP_AV                   |
                IBV_QP_PATH_MTU             |
                IBV_QP_DEST_QPN             |
                IBV_QP_RQ_PSN               |
                IBV_QP_MAX_DEST_RD_ATOMIC   |
                IBV_QP_MIN_RNR_TIMER),
        "Could not modify QP to RTR state");
    }
    //free(attr);
}

static void qp_change_state_rts()
{
    //struct ibv_qp_attr *attr;
    //attr =  malloc(sizeof *attr);
    //memset(attr, 0, sizeof *attr);

    struct ibv_qp_attr attrPool[100];
    for(int ii=0;ii<ctx->nQPair;ii++)
    {
      struct ibv_qp_attr *attr=&attrPool[ii];
      memset(attr, 0, sizeof *attr);
      attr->qp_state              = IBV_QPS_RTS;
      attr->timeout               = 14;
      attr->retry_cnt             = 7;
      attr->rnr_retry             = 7;    /* infinite retry */
      attr->sq_psn                = ctx->lConn.psnV[ii];
      attr->max_rd_atomic         = 1;
     
      TEST_NZ(ibv_modify_qp(ctx->qp[ii], attr,
                IBV_QP_STATE            |
                IBV_QP_TIMEOUT          |
                IBV_QP_RETRY_CNT        |
                IBV_QP_RNR_RETRY        |
                IBV_QP_SQ_PSN           |
                IBV_QP_MAX_QP_RD_ATOMIC),
        "Could not modify QP to RTS state");
    }
    //free(attr);
}

void query_qp()
{
  struct ibv_qp *qp;
  struct ibv_qp_attr attr;
  struct ibv_qp_init_attr init_attr;
 
  for(int ii=0;ii<ctx->nQPair;ii++)
  {
    if (ibv_query_qp(ctx->qp[ii], &attr, IBV_QP_STATE, &init_attr))
    {
	fprintf(stderr, "Failed to query QP state\n");
        return;
    }
    //printf("qp id=%d in state %s (%d)\n",ii,ibv_qp_status_str(attr.qp_state),attr.qp_state);
    printf("qp id=%d in state (%d)\n",ii,attr.qp_state);
  }
}
//static void destroy_ctx(struct app_context *ctx)
void destroy_ib()
{
     
     for(int ii=0;ii<ctx->nQPair;ii++)
     {
        //printf("destroying qp %d\n",ii);

	TEST_NZ(ibv_destroy_qp(ctx->qp[ii]),
		"Could not destroy queue pair, ibv_destroy_qp");
     }
        free(ctx->qp);
        free(ctx->nMsgQ);
	
     for(int ii=0;ii<ctx->ncq;ii++)
	TEST_NZ(ibv_destroy_cq(ctx->cq[ii]),
		"Could not destroy send completion queue, ibv_destroy_cq");

        free(ctx->cq);
	
	//TEST_NZ(ibv_destroy_comp_channel(ctx->ch),
        //		"Could not destory completion channel, ibv_destroy_comp_channel");

	TEST_NZ(ibv_dereg_mr(ctx->mr),
		"Could not de-register memory region, ibv_dereg_mr");

	TEST_NZ(ibv_dealloc_pd(ctx->pd),
		"Could not deallocate protection domain, ibv_dealloc_pd");	

	//free(ctx->buf); //will be freed by the driver
        free(ctx->lConn.qpnV);
        free(ctx->lConn.psnV);
        free(ctx->rConn);
	free(ctx);
	
}

static int die(const char *reason){
	fprintf(stderr, "Err: %s - %s\n ", strerror(errno), reason);
	exit(EXIT_FAILURE);
	return -1;
}
