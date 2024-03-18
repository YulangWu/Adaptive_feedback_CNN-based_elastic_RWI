!********************************************************************
!
! program forward.f90
!
!********************************************************************
!
!   Author:       Yulang Wu & Bao D. Nguyen
!   Date:         Fall 2017
!   Copyright:    Center for Lithospheric Studies,
!                 The University of Texas at Dallas
!   Environment:  
!       1. Workstation information
!           1.1 Operating System; Name: CentOS Linux; Release: 6.7
!           1.2 CPU: Number of sockets: 2;  
!                    Cores per socket: 4;
!                    Threads per cores: 2     
!               CPU type: Intel(R) Xeon(R) E5630 CPU @ 2.53 GHz; 
!               Effective number of threads: 16
!           1.3 Installed Memory: 23 GB
!
!********************************************************************
!
! "Include" file:   
!
!   globalvar.f90         : Sets variables that are consistent across
!                           all programs and inputs parameters
!
!********************************************************************
!
! Subroutines:
!
!   pml.f90           : Defines the perfectly-matched layer (PML) 
!                       absorbing boundaries at all four edges
!   stepvelstress.f90 : Numerically solves the first-order velocity 
!                       -stress equations
!********************************************************************

program inversion_data 
implicit none !No undeclared variables are allowed
integer::water_depth,total_iteration
include "./common/globalvar.f90" !include common global variables

print *,"Input the depth of water:"
read(*,*)water_depth

print *,"Input the number of iterations:"
read(*,*)total_iteration

call system('mkdir output_data')
do shot_num = 0,total_iteration
    image_vz = 0.d0
    image_vx = 0.d0
    print *,'process shot = ',shot_num
    !--------------------------------------------------------------------
    !
    ! input the true vp model:
    !
    vp(:,:)  = real0    
    vs(:,:)  = real0    
    rho(:,:)  = real0    

    write(filenumber_iteration,'(i5)')shot_num
    open(3, file='../matlab1.0/models/'//trim(adjustl(filenumber_iteration))//'th_true_vp.dat',status='old'& 
    ,form='formatted', access='direct', recl=17,iostat=rc)
    if(rc/=0) print *,'open file failed!'
    do i = 1,nx
    do j = 1,nz
        read(3,'(f17.8)',rec=j+(i-1)*nz) vp(j,i) 
    end do
    end do
    close(3)

    do i = -2,0
        vp(:,i) = vp(:,1)
        vp(i,:) = vp(1,:)
    end do

    do i = 1,3
        vp(:,nx+i)=vp(:,nx)
        vp(nz+i,:)=vp(nz,:)
    end do


    open(3, file='../matlab1.0/models/'//trim(adjustl(filenumber_iteration))//'th_true_vs.dat',status='old'& 
    ,form='formatted', access='direct', recl=17,iostat=rc)
    if(rc/=0) print *,'open file failed!'
    do i = 1,nx
    do j = 1,nz
        read(3,'(f17.8)',rec=j+(i-1)*nz) vs(j,i) 
    end do
    end do
    close(3)

    do i = -2,0
        vs(:,i) = vs(:,1)
        vs(i,:) = vs(1,:)
    end do

    do i = 1,3
        vs(:,nx+i)=vs(:,nx)
        vs(nz+i,:)=vs(nz,:)
    end do



    open(3, file='../matlab1.0/models/'//trim(adjustl(filenumber_iteration))//'th_true_rho.dat',status='old'& 
    ,form='formatted', access='direct', recl=17,iostat=rc)
    if(rc/=0) print *,'open file failed!'
    do i = 1,nx
    do j = 1,nz
        read(3,'(f17.8)',rec=j+(i-1)*nz) rho(j,i) 
    end do
    end do
    close(3)

    do i = -2,0
        rho(:,i) = rho(:,1)
        rho(i,:) = rho(1,:)
    end do

    do i = 1,3
        rho(:,nx+i)=rho(:,nx)
        rho(nz+i,:)=rho(nz,:)
    end do

    vp(1:water_depth,:)=1.5
    vs(1:water_depth,:)=0
    rho(1:water_depth,:)=1.01
    vs = vs*1000;
    vp = vp*1000;
    rho = rho*1000;

    ! vs = 0.d0;
    ! rho = 1000.d0



    !--------------------------------------------------------------------
    temp = dt * sqrt(1.d0/dz**2 + 1.d0/dx**2)
    do ix = -2,nx+3
        do iz = -2,nz+3
            CourantNum = real0
            CourantNum = vp(iz,ix) * temp
            if(CourantNum > 0.6d0) stop 'dt too large, simulation will be unstable'
        end do
    end do


    ! Compute Lame parameters:
    mu(:,:)     = rho(:,:)*vs(:,:)*vs(:,:)
    lambda(:,:) = rho(:,:)*(vp(:,:)*vp(:,:) - 2.*vs(:,:)*vs(:,:))
    
    vp_mig = vp(1,1)
    rho_mig = rho(1,1)
    ! Compute Lame parameters:
    mu_mig(:,:)     = rho_mig(:,:)*vs_mig(:,:)*vs_mig(:,:)
    lambda_mig(:,:) = rho_mig(:,:)*(vp_mig(:,:)*vp_mig(:,:) - 2.*vs_mig(:,:)*vs_mig(:,:))


    !do ixs = nx/2,nx/2 !5,nx-4,incre_s
        print *, ixs
        !--------------------------------------------------------------------
        ! Define PML parameters:
        call define_pml(nz, nx, npml, vp, f0, dz, dx, dt, a_z, b_z,  &
        K_z, a_z_half, b_z_half, K_z_half, a_x, b_x, K_x, a_x_half,  &
        b_x_half, K_x_half)

        

        !********************************************************************
        ! 1. Forward wavefield modeling with direct wave
        !********************************************************************
        print *,'1. Forward propagation with direct wave:'
        !--------------------------------------------------------------------
        ! Array initializations:

        ! Stresses & particle velocities

        vz(:,:)  = real0    
        vx(:,:)  = real0
        szz(:,:) = real0
        sxx(:,:) = real0
        sxz(:,:) = real0

        ! Wavefield temporary variables to solve elastodynamic equations
        mem_dvz_dz(:,:)  = real0    
        mem_dvz_dx(:,:)  = real0
        mem_dvx_dz(:,:)  = real0
        mem_dvx_dx(:,:)  = real0
        mem_dszz_dz(:,:) = real0
        mem_dsxx_dx(:,:) = real0
        mem_dsxz_dz(:,:) = real0
        mem_dsxz_dx(:,:) = real0

        !--------------------------------------------------------------------

        itt = 0
        do it=1,nt  !Loop over time
            
            ! if(mod(it,it_display) == 0) print *, it
        !   Compute stresses and particle velocities:
            call stepvelstress(npml,lambda, mu, rho, nz, nx, aa, vz, vx, dz,  &
                dx, dt, szz, sxx, sxz, mem_dvz_dz, mem_dvz_dx, mem_dvx_dz,  &
                mem_dvx_dx, mem_dszz_dz, mem_dsxx_dx, mem_dsxz_dz,        &
                mem_dsxz_dx, a_z, b_z, K_z, a_z_half, b_z_half, K_z_half,  &
                a_x, b_x, K_x, a_x_half, b_x_half, K_x_half)
            
            if(mod(it,it_display) == -1)then
                write(filenumber_iteration,'(i5)')it
                filename = './output_data/forwavefield_vz'//trim(adjustl(filenumber_iteration))//'.dat'
                open(110, file=filename)
                close(110,status='delete')
                open(110, file=filename,status='new'& 
                ,form='formatted', access='direct', recl=17,iostat=rc)
                if(rc/=0) print *,'open file failed!'

                write(filenumber_iteration,'(i5)')it
                filename = './output_data/forwavefield_vx'//trim(adjustl(filenumber_iteration))//'.dat'
                open(111, file=filename)
                close(111,status='delete')
                open(111, file=filename,status='new'& 
                ,form='formatted', access='direct', recl=17,iostat=rc)
                if(rc/=0) print *,'open file failed!'

                
                do i = -2,nx+3
                    do j = -2,nz+3
                        write(110,'(f17.8)',rec=(j+2)+1+(i+2)*(nz+6)) vz(j,i) 
                        write(111,'(f17.8)',rec=(j+2)+1+(i+2)*(nz+6)) vx(j,i) 
                    end do
                end do
                close(110)
                close(111)
                
            end if
            
            !store the source wavefield
            if(mod(it,incre_t) == 0)then
            
                itt = itt + 1
                vz_wavefield(1:nz,1:nx,itt) = vz(1:nz,1:nx)
                vx_wavefield(1:nz,1:nx,itt) = vx(1:nz,1:nx)
            end if
            
            !store the seimogram data 
            vz_seismo(1:nx,it) = vz(izs,1:nx) 
            vx_seismo(1:nx,it) = vx(izs,1:nx) 

        !   Add source term to normal stresss sxx and szz:
            t = real(it-1)*dt

        !   First derivative of Gaussian:
            source_term(it) = -factor*2.*a*(t-t0)*exp(-a*(t-t0)**2)
            
            !do loop for plane sources
            do ixs = 5,nx-4,2 !incre_s
              szz(izs,ixs)    = szz(izs,ixs) + source_term(it)
              sxx(izs,ixs)    = sxx(izs,ixs) + source_term(it)
            end do
            
        end do


        ! !********************************************************************
        ! ! 2. Forward wavefield modeling create direct wave only
        ! !********************************************************************
        ! print *,'1. Forward propagation without direct wave:'
        ! !--------------------------------------------------------------------
        ! ! Array initializations:
        ! ! Stresses & particle velocities

        ! vz(:,:)  = real0    
        ! vx(:,:)  = real0
        ! szz(:,:) = real0
        ! sxx(:,:) = real0
        ! sxz(:,:) = real0

        ! ! Wavefield temporary variables to solve elastodynamic equations
        ! mem_dvz_dz(:,:)  = real0    
        ! mem_dvz_dx(:,:)  = real0
        ! mem_dvx_dz(:,:)  = real0
        ! mem_dvx_dx(:,:)  = real0
        ! mem_dszz_dz(:,:) = real0
        ! mem_dsxx_dx(:,:) = real0
        ! mem_dsxz_dz(:,:) = real0
        ! mem_dsxz_dx(:,:) = real0

        ! !--------------------------------------------------------------------
        
        ! do it=1,nt  !Loop over time
        !     ! if(mod(it,it_display) == 0) print *, it
        !     !   Compute stresses and particle velocities:
        !     call stepvelstress(npml,lambda_mig, mu_mig, rho_mig, nz, nx, aa, vz, vx, dz,  &
        !         dx, dt, szz, sxx, sxz, mem_dvz_dz, mem_dvz_dx, mem_dvx_dz,  &
        !         mem_dvx_dx, mem_dszz_dz, mem_dsxx_dx, mem_dsxz_dz,        &
        !         mem_dsxz_dx, a_z, b_z, K_z, a_z_half, b_z_half, K_z_half,  &
        !         a_x, b_x, K_x, a_x_half, b_x_half, K_x_half)

        !     if(mod(it,it_display) == -1)then
        !         write(filenumber_iteration,'(i5)')it
        !         filename = './output_inversion/wavefield_vz_Dwave_only'//trim(adjustl(filenumber_iteration))//'.dat'
        !         open(110, file=filename)
        !         close(110,status='delete')
        !         open(110, file=filename,status='new'& 
        !         ,form='formatted', access='direct', recl=17,iostat=rc)
        !         if(rc/=0) print *,'open file failed!'

        !         write(filenumber_iteration,'(i5)')it
        !         filename = './output_inversion/wavefield_vx_Dwave_only'//trim(adjustl(filenumber_iteration))//'.dat'
        !         open(111, file=filename)
        !         close(111,status='delete')
        !         open(111, file=filename,status='new'& 
        !         ,form='formatted', access='direct', recl=17,iostat=rc)
        !         if(rc/=0) print *,'open file failed!'

                
        !         do i = -2,nx+3
        !             do j = -2,nz+3
        !                 write(110,'(f17.8)',rec=(j+2)+1+(i+2)*(nz+6)) vz(j,i) 
        !                 write(111,'(f17.8)',rec=(j+2)+1+(i+2)*(nz+6)) vx(j,i) 
        !             end do
        !         end do
        !         close(110)
        !         close(111)
        !     end if

        !     !remove the direct wave from the seimogram data 
        !     vz_seismo(1:nx,it) = vz_seismo(1:nx,it)  - vz(izs,1:nx) 
        !     vx_seismo(1:nx,it) = vx_seismo(1:nx,it)  - vx(izs,1:nx) 

        ! !   Add source term to normal stresss sxx and szz:
        !     t = real(it-1)*dt

        ! !   First derivative of Gaussian:
        !     source_term(it) = -factor*2.*a*(t-t0)*exp(-a*(t-t0)**2)
        !     szz(izs,ixs)    = szz(izs,ixs) + source_term(it)
        !     sxx(izs,ixs)    = sxx(izs,ixs) + source_term(it)
        ! end do
        
    !end do !ixs loop


    write(filenumber_iteration,'(i5)')shot_num
    open(10, file='./output_data/seis_data'//trim(adjustl(filenumber_iteration))//'.dat')
    close(10,status='delete')
    open(10, file='./output_data/seis_data'//trim(adjustl(filenumber_iteration))//'.dat',status='new'& 
    ,form='formatted', access='direct', recl=17,iostat=rc)
    if(rc/=0) print *,'open file failed!'

    do i=1,nt
    do j=1,nx
            write(10,'(f17.8)',rec=j + (i-1)*nx + 0*nx*nt) vz_seismo(j,i)/100
    end do
    end do

    do i=1,nt
    do j=1,nx
            write(10,'(f17.8)',rec=j + (i-1)*nx + 1*nx*nt) vx_seismo(j,i)/100
    end do
    end do

    close(10)
end do

! call exit
end program inversion_data

include "./common/stepvelstress.f90"
include "./common/pml.f90"
!********************************************************************
! Main Program End
!********************************************************************
