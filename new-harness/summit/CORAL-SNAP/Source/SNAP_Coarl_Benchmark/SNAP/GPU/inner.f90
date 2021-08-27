!-----------------------------------------------------------------------
!
! MODULE: inner_module
!> @brief
!> This module controls the inner iterations. Inner iterations include
!> the KBA mesh sweep, which is parallelized via MPI and vectorized over
!> angles in a given octant. Inner source computed here and inner
!> convergence is checked.
!
!-----------------------------------------------------------------------

MODULE inner_module

  USE global_module, ONLY: i_knd, r_knd, l_knd, zero, one, ounit

  USE geom_module, ONLY: nx, ny, nz

  USE sn_module, ONLY: nmom, cmom, lma, nang

  USE data_module, ONLY: ng

  USE control_module, ONLY: epsi, tolr, dfmxi, inrdone, it_det, use_gpu

  USE solvar_module, ONLY: q2grp, s_xs, flux, fluxpi, fluxm, qtot, h_fluxm

  USE sweep_module, ONLY: sweep

  USE time_module, ONLY: tinrsrc, tsweeps, wtime, tinrconv, inrconv12,inrconv23,inrconv34,inrconv45

  USE plib_module, ONLY: glmax, comm_snap, iproc, root, yproc, zproc

  IMPLICIT NONE

  PRIVATE

  PUBLIC :: inner


  CONTAINS


  SUBROUTINE inner ( inno, iits )

!-----------------------------------------------------------------------
!
! Do a single inner iteration for all groups. Calculate the total source
! for each group and sweep the mesh.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: inno

    INTEGER(i_knd), DIMENSION(ng), INTENT(OUT) :: iits

    REAL(r_knd) :: t1, t2, t3, t4, t5
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g
!_______________________________________________________________________
!
!   Compute the inner source and add it to fixed + out-of-group sources
!_______________________________________________________________________

    CALL wtime ( t1 )

    CALL inr_src

    CALL wtime ( t2 )
    tinrsrc = tinrsrc + t2 - t1
!_______________________________________________________________________
!
!   With source computed, set previous copy of flux and zero out current
!   copies--new flux moments iterates computed during sweep. Thread
!   over groups.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      IF ( use_gpu .EQV. .FALSE. ) THEN
        fluxpi(:,:,:,g)   = flux(:,:,:,g)
      ELSE 
        CALL copy_fluxpi ( nx, ny, nz, g )
      END IF
    END DO
  !$OMP END PARALLEL DO
  !CALL zero_flux ( cmom, nx, ny, nz, ng )
!_______________________________________________________________________
!
!   Call for the transport sweep. Check convergence, using threads.
!_______________________________________________________________________

    CALL wtime ( t3 )

    CALL sweep 

!    CALL dump_flux(nx,ny,nz,inno,inno)
!    CALL dump_fluxm(nx,ny,nz,cmom,inno,inno)

    CALL wtime ( t4 )
    tsweeps = tsweeps + t4 - t3

    CALL inr_conv ( inno, iits )
    CALL wtime ( t5 )
    tinrconv = tinrconv + t5 - t4
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inner


  SUBROUTINE inr_src

!-----------------------------------------------------------------------
!
! Compute the inner source, i.e., the within-group scattering source.
!
!-----------------------------------------------------------------------
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g
!_______________________________________________________________________
!
!   Compute the within-group scattering source. Thread over groups.
!_______________________________________________________________________

  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      IF ( use_gpu .EQV. .FALSE. ) THEN
        CALL inr_src_scat ( q2grp(:,:,:,:,g), s_xs(:,:,:,:,g),           &
          flux(:,:,:,g), fluxm(:,:,:,:,g), qtot(:,:,:,:,g) )
      ELSE
        CALL inr_src_scat_cuda ( nang, cmom, nmom, nx, ny, nz, g )
      END IF
    END DO
  !$OMP END PARALLEL DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_src


  SUBROUTINE inr_src_scat ( qo, cs, f, fm, q )

!-----------------------------------------------------------------------
!
! Compute the within-group scattering for a given group. Add it to fixed
! and out-of-group sources.
!
!-----------------------------------------------------------------------

    REAL(r_knd), DIMENSION(nx,ny,nz), INTENT(IN) :: f

    REAL(r_knd), DIMENSION(cmom-1,nx,ny,nz), INTENT(IN) :: fm

    REAL(r_knd), DIMENSION(cmom,nx,ny,nz), INTENT(IN) :: qo

    REAL(r_knd), DIMENSION(nmom,nx,ny,nz), INTENT(IN) :: cs

    REAL(r_knd), DIMENSION(cmom,nx,ny,nz), INTENT(OUT) :: q
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: i, j, k, l, m, mom
!_______________________________________________________________________
!
!   Loop over all cells. Set the first source moment with flux (f). Then
!   set remaining source moments with fluxm (fm) and combination of
!   higher scattering orders.
!_______________________________________________________________________

    DO k = 1, nz
    DO j = 1, ny
    DO i = 1, nx

      q(1,i,j,k) = qo(1,i,j,k) + cs(1,i,j,k)*f(i,j,k)
!_______________________________________________________________________
!
!     Work on other moments with fluxm array
!_______________________________________________________________________

      mom = 2
      DO l = 2, nmom
        DO m = 1, lma(l)
          q(mom,i,j,k) = qo(mom,i,j,k) + cs(l,i,j,k)*fm(mom-1,i,j,k)
          mom = mom + 1
        END DO
      END DO

    END DO
    END DO
    END DO
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_src_scat


  SUBROUTINE inr_conv ( inno, iits )

!-----------------------------------------------------------------------
!
! Check for inner iteration convergence using the flux array.
!
!-----------------------------------------------------------------------

    INTEGER(i_knd), INTENT(IN) :: inno

    INTEGER(i_knd), DIMENSION(ng), INTENT(OUT) :: iits
!_______________________________________________________________________
!
!   Local variables
!_______________________________________________________________________

    INTEGER(i_knd) :: g

    REAL(r_knd), DIMENSION(nx,ny,nz,ng) :: df
    REAL(r_knd) :: t1, t2, t3, t4, t5
!_______________________________________________________________________
!
!   Thread group loops for computing local difference (df) array.
!   compute max for that group.
!_______________________________________________________________________

    CALL wtime ( t1 )
  if (use_gpu) call d2h_flux(flux,h_fluxm,cmom,nx,ny,nz,ng)

!  write (*,*) 'flux=',flux(1:3,1:3,1:3,1:3)
!  write (*,*) 'fluxg=',flux(1,1,1,:)
!  write (*,*) 'fluxm1=',h_fluxm(1:3,1:3,1:3,1,1:3)
!  write (*,*) 'fluxm2=',h_fluxm(1:3,1:3,1:3,2,1:3)
!  write (*,*) 'fluxm3=',h_fluxm(1:3,1:3,1:3,3,1:3)
!  write (*,*) 'fluxm=',fluxm(1,2,1,1,1)

!  print *, flux(:,:,:,ng)

    CALL wtime ( t2 )
  !$OMP PARALLEL DO SCHEDULE(DYNAMIC,1) DEFAULT(SHARED) PRIVATE(g)
    DO g = 1, ng
      IF ( inrdone(g) ) CYCLE
      iits(g) = inno
      IF ( use_gpu .EQV. .FALSE. ) THEN
        WHERE( ABS( fluxpi(:,:,:,g) ) > tolr )
          df(:,:,:,g) = ABS( flux(:,:,:,g)/fluxpi(:,:,:,g) - one )
        ELSEWHERE
          df(:,:,:,g) = ABS( flux(:,:,:,g) - fluxpi(:,:,:,g) )
        END WHERE
        dfmxi(g) = MAXVAL( df(:,:,:,g) )
      ELSE
        CALL compute_df_cuda ( nx, ny, nz, g, tolr, dfmxi )
      END IF 
    END DO
  !$OMP END PARALLEL DO
    CALL wtime ( t3 )
!_______________________________________________________________________
!
!   All procs then reduce dfmxi for all groups, determine which groups
!   are converged and print requested info
!_______________________________________________________________________

    CALL glmax ( dfmxi, ng, comm_snap )
    CALL wtime ( t4 )
!    if (iproc==root) then
!       print *, sum(dfmxi(1:ng)), dfmxi(1), dfmxi(ng)
!    endif
    WHERE( dfmxi <= epsi ) inrdone = .TRUE.
    IF ( iproc==root .AND. it_det==1 ) THEN
      DO g = 1, ng
        WRITE( ounit, 221 ) g, iits(g), dfmxi(g)
      END DO
    END IF
    CALL wtime ( t5 )
    inrconv12 = inrconv12 + t2 - t1
    inrconv23 = inrconv23 + t4 - t2
    inrconv34 = inrconv34 + t5 - t3
    inrconv45 = inrconv45 + t5 - t4
!_______________________________________________________________________

    221 FORMAT( 4X, 'Group ', I3, 4X, ' Inner ', I5, 4X, ' Dfmxi ',    &
                ES11.4 )
!_______________________________________________________________________
!_______________________________________________________________________

  END SUBROUTINE inr_conv


END MODULE inner_module
