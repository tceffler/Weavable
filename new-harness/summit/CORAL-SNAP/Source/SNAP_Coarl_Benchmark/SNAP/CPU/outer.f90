!-----------------------------------------------------------------------
!
! MODULE: outer_module
!> @brief
!> This module controls the outer iterations. Outer iterations are
!> threaded over the energy dimension and represent a Jacobi iteration
!> strategy. Includes setting the outer source. Checking outer iteration
!> convergence.
!
!-----------------------------------------------------------------------

MODULE outer_module

  USE global_module, ONLY: i_knd, r_knd, zero, one

  USE geom_module, ONLY: nx, ny, nz

  USE sn_module, ONLY: nmom, cmom, lma

  USE data_module, ONLY: ng, mat, slgg, qi, nmat, src_opt

  USE solvar_module, ONLY: q2grp, flux, fluxm, fluxpo

  USE control_module, ONLY: iitm, timedep, inrdone, tolr, dfmxo, epsi, &
    otrdone

  USE inner_module, ONLY: inner

  USE time_module, ONLY: totrsrc, tinners, tinrmisc, wtime

  USE plib_module, ONLY: glmax, comm_snap

  USE expxs_module, ONLY: expxs_reg

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: outer


  CONTAINS


  SUBROUTINE outer ( sum_iits )

!-----------------------------------------------------------------------
!
! Do a single outer iteration. Sets the out-of-group sources, performs
! inners for all groups.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(OUT) :: sum_iits
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g, inno

    INTEGER(i_knd), DIMENSION(ng) :: iits

    REAL(r_knd) :: t1, t2, t3, t4
!_______________________________________________________________________
!
!   Compute the outer source: sum of fixed + out-of-group sources
!_______________________________________________________________________

    CALL wtime ( t1 )

    CALL otr_src

    CALL wtime ( t2 )
    totrsrc = totrsrc + t2 - t1
!_______________________________________________________________________
!
!   Zero out the inner iterations group count. Save the flux for
!   comparison. Parallelize group loop with threads.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1)
    DO g = 1, ng
      iits(g) = 0
      fluxpo(:,:,:,g) = flux(:,:,:,g)
    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!
!   Start the inner iterations
!_______________________________________________________________________

    CALL wtime ( t3 )

    inrdone = .FALSE.

    inner_loop: DO inno = 1, iitm

      CALL inner ( inno, iits )

      IF ( ALL( inrdone ) ) EXIT inner_loop

    END DO inner_loop

    sum_iits = SUM( iits )

    CALL wtime ( t4 )
    tinners = tinners + t4 - t3
!_______________________________________________________________________
!
!   Check outer convergence
!_______________________________________________________________________

    CALL otr_conv
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE outer


  SUBROUTINE otr_src

!-----------------------------------------------------------------------
!
! Loop over groups to compute each one's outer loop source.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local Variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g, gp
!_______________________________________________________________________
!
!   Initialize the source to fixed. Parallelize outer group loop with
!   threads.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED)                &
  !$OMP& PRIVATE(g,gp) 
    DO g = 1, ng

      q2grp(1,:,:,:,g) = qi(:,:,:,g)
!_______________________________________________________________________
!
!     Loop over cells and moments to compute out-of-group scattering
!_______________________________________________________________________

      DO gp = 1, ng
        IF ( gp == g ) CYCLE
        CALL otr_src_scat ( q2grp(:,:,:,:,g), slgg(:,:,gp,g),      &
          mat, flux(:,:,:,gp), fluxm(:,:,:,:,gp) )
      END DO

    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE otr_src


  SUBROUTINE otr_src_scat ( q, cs, map, f, fm )

!-----------------------------------------------------------------------
!
! Compute the scattering source for all cells and moments
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: map

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: f

    REAL(r_knd), DIMENSION(cmom-1,nx,ny,nz), INTENT(IN) :: fm

    REAL(r_knd), DIMENSION(nmat,nmom), INTENT(IN) :: cs

    REAL(r_knd), DIMENSION(cmom,nx,ny,nz), INTENT(INOUT) :: q
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: l, m, mom, i, j, k, m1, mbeg(nmom)

!YKTREAL(r_knd), DIMENSION(nx,ny,nz) :: tc
    REAL(r_knd) :: tc
!_______________________________________________________________________
!
!   Use expxs_reg to expand current gp->g cross sections to the fine
!   mesh one scattering order at a time. Start with first. Then compute
!   source moment with flux.
!_______________________________________________________________________

!YKT : just use the value for tc that would be returned by expxs_reg()
!YKTCALL expxs_reg ( cs(:,1), map, tc )

!YKTq(1,:,:,:) = q(1,:,:,:) + tc*f
    do k = 1, nz
    do j = 1, ny
    do i = 1, nx
      q(1,i,j,k) = q(1,i,j,k) + cs(map(i,j,k),1)*f(i,j,k)
    end do
    end do
    end do
!_______________________________________________________________________
!
!   Repeat the process for higher scattering orders, source moments.
!   Use loop for multiple source moments of same scattering order.
!_______________________________________________________________________

    mom = 2
    do l = 2, nmom
      mbeg(l) = mom
      do m = 1, lma(l)
        mom = mom + 1
      end do
    end do

    do k = 1, nz
    do j = 1, ny
      DO l = 2, nmom
        m1 = mbeg(l) - 1
!YKT    CALL expxs_reg ( cs(:,l), map, tc )
        do i = 1, nx
          tc = cs(map(i,j,k),l)
          DO m = 1, lma(l)
            q(m1+m,i,j,k) = q(m1+m,i,j,k) + tc*fm(m1+m-1,i,j,k)
          END DO
        end do
      END DO
    end do
    end do
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE otr_src_scat


  SUBROUTINE otr_conv

!-----------------------------------------------------------------------
!
! Check for convergence of outer iterations.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g

    REAL(r_knd), DIMENSION(nx,ny,nz,ng) :: df
!_______________________________________________________________________
!
!   Thread to speed up computation of df by looping over groups. Rejoin
!   threads and then determine max error.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      WHERE( ABS( fluxpo(:,:,:,g) ) > tolr )
        df(:,:,:,g) = ABS( flux(:,:,:,g)/fluxpo(:,:,:,g) - one )
      ELSEWHERE
        df(:,:,:,g) = ABS( flux(:,:,:,g) - fluxpo(:,:,:,g) )
      END WHERE
    END DO
  !$OMP END PARALLEL DO

    dfmxo = MAXVAL( df )
    CALL glmax ( dfmxo, comm_snap )

    IF ( dfmxo<=100.0_r_knd*epsi .AND. ALL( inrdone ) ) otrdone = .TRUE.
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE otr_conv


END MODULE outer_module
