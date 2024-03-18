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

program migration 
implicit none !No undeclared variables are allowed
include "./common/globalvar.f90" !include common global variables

call system('mkdir output_inversion')
do shot_num = 16,21
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
    open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(truevp))//'vp.dat',status='old'& 
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


    ! open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(truevp))//'vs.dat',status='old'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = 1,nx
    ! do j = 1,nz
    !     read(3,'(f17.8)',rec=j+(i-1)*nz) vs(j,i) 
    ! end do
    ! end do
    ! close(3)

    ! do i = -2,0
    !     vs(:,i) = vs(:,1)
    !     vs(i,:) = vs(1,:)
    ! end do

    ! do i = 1,3
    !     vs(:,nx+i)=vs(:,nx)
    !     vs(nz+i,:)=vs(nz,:)
    ! end do



    ! open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(truevp))//'rho.dat',status='old'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = 1,nx
    ! do j = 1,nz
    !     read(3,'(f17.8)',rec=j+(i-1)*nz) rho(j,i) 
    ! end do
    ! end do
    ! close(3)

    ! do i = -2,0
    !     rho(:,i) = rho(:,1)
    !     rho(i,:) = rho(1,:)
    ! end do

    ! do i = 1,3
    !     rho(:,nx+i)=rho(:,nx)
    !     rho(nz+i,:)=rho(nz,:)
    ! end do


    vs = vs*1000;
    vp = vp*1000;
    rho = rho*1000;



    !
    ! input the true vp model:
    !
    vp_init(:,:)  = real0   
    vs_init(:,:)  = real0    
    rho_init(:,:)  = real0    

    open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(initvp))//'vp.dat',status='old'& 
    ,form='formatted', access='direct', recl=17,iostat=rc)
    if(rc/=0) print *,'open file failed!'
    do i = 1,nx
    do j = 1,nz
        read(3,'(f17.8)',rec=j+(i-1)*nz) vp_init(j,i) 
    end do
    end do
    close(3)

    do i = -2,0
        vp_init(:,i) = vp_init(:,1)
        vp_init(i,:) = vp_init(1,:)
    end do

    do i = 1,3
        vp_init(:,nx+i)=vp_init(:,nx)
        vp_init(nz+i,:)=vp_init(nz,:)
    end do


    ! open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(initvp))//'vs.dat',status='old'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = 1,nx
    ! do j = 1,nz
    !     read(3,'(f17.8)',rec=j+(i-1)*nz) vs_init(j,i) 
    ! end do
    ! end do
    ! close(3)

    ! do i = -2,0
    !     vs_init(:,i) = vs_init(:,1)
    !     vs_init(i,:) = vs_init(1,:)
    ! end do
    ! +i)=vs_init(:,nx)
    !     vs_init(nz+i,:)=vs_init(nz,:)
    ! end do+i)=vs_init(:,nx)
    !     vs_init(nz+i,:)=vs_init(nz,:)
    ! end do


    ! open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(initvp))//'rho.dat',status='old'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = 1,nx
    ! do j = 1,nz
    !     read(3,'(f17.8)',rec=j+(i-1)*nz) rho_init(j,i) 
    ! end do
    ! end do
    ! close(3)

    ! do i = -2,0
    !     rho_init(:,i) = rho_init(:,1)
    !     rho_init(i,:) = rho_init(1,:)
    ! end do

    ! do i = 1,3
    !     rho_init(:,nx+i)=rho_init(:,nx)
    !     rho_init(nz+i,:)=rho_init(nz,:)
    ! end do

    ! open(3, file='./given_models_inversion/'//trim(adjustl(filenumber_iteration))//trim(adjustl(truevp))//'rho.dat',status='old'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = 1,nx
    ! do j = 1,nz
    !     read(3,'(f17.8)',rec=j+(i-1)*nz) rho(j,i) 
    ! end do
    ! end do
    ! close(3)

    ! do i = -2,0
    !     rho(:,i) = rho(:,1)
    !     rho(i,:) = rho(1,:)
    ! end do

    ! do i = 1,3
    !     rho(:,nx+i)=rho(:,nx)
    !     rho(nz+i,:)=rho(nz,:)
    ! end do
    vs_init = vs_init*1000;
    vp_init = vp_init*1000;
    rho_init = rho_init*1000;

    vs = 0.d0;
    rho = 1000.d0
    vs_init = 0.d0;
    rho_init =1000.d0

    ! open(4, file='./output_inversion/vp.dat')
    ! close(4,status='delete')
    ! open(4, file='./output_inversion/vp_init.dat',status='new'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = -2,nx+3
    ! do j = -2,nz+3
    !     write(4,'(f17.8)',rec=(j+2)+1+(i+2)*(nz+6)) vp_init(j,i) 
    ! end do
    ! end do
    ! close(4)

    ! open(5, file='./output_inversion/rho_init.dat')
    ! close(5,status='delete')
    ! open(5, file='./output_inversion/rho.dat',status='new'& 
    ! ,form='formatted', access='direct', recl=17,iostat=rc)
    ! if(rc/=0) print *,'open file failed!'
    ! do i = -2,nx+3
    ! do j = -2,nz+3
    !     write(5,'(f17.8)',rec=(j+2)+1+(i+2)*(nz+6)) rho_init(j,i) 
    ! end do
    ! end do
    ! close(5)

    !--------------------------------------------------------------------
    temp = dt * sqrt(1.d0/dz**2 + 1.d0/dx**2)
    do ix = -2,nx+3
        do iz = -2,nz+3
            CourantNum = real0
            CourantNum = vp(iz,ix) * temp
            if(CourantNum > 0.6d0) stop 'dt too large, simulation will be unstable'
        end do
    end do




    !--------------------------------------------------------------------
    ! Compute Lame parameters:
    mu(:,:)     = rho(:,:)*vs(:,:)*vs(:,:)
    lambda(:,:) = rho(:,:)*(vp(:,:)*vp(:,:) - 2.*vs(:,:)*vs(:,:))

    vp_mig = vp(1,1)
    rho_mig = rho(1,1)
    ! Compute Lame parameters:
    mu_mig(:,:)     = rho_mig(:,:)*vs_mig(:,:)*vs_mig(:,:)
    lambda_mig(:,:) = rho_mig(:,:)*(vp_mig(:,:)*vp_mig(:,:) - 2.*vs_mig(:,:)*vs_mig(:,:))

    ! Compute Lame parameters:
    mu_init(:,:)     = rho_init(:,:)*vs_init(:,:)*vs_init(:,:)
    lambda_init(:,:) = rho_init(:,:)*(vp_init(:,:)*vp_init(:,:) - 2.*vs_init(:,:)*vs_init(:,:))

    do ixs = 5,nx-4,incre_s
        print *, ixs
        !********************************************************************
        ! 1. Forward wavefield modeling with direct wave
        !********************************************************************
        !--------------------------------------------------------------------
        ! Define PML parameters:
        call define_pml(nz, nx, npml, vp, f0, dz, dx, dt, a_z, b_z,  &
        K_z, a_z_half, b_z_half, K_z_half, a_x, b_x, K_x, a_x_half,  &
        b_x_half, K_x_half)

        print *,'1. Forward propagation:'


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
                filename = './output_inversion/forwavefield_vz'//trim(adjustl(filenumber_iteration))//'.dat'
                open(110, file=filename)
                close(110,status='delete')
                open(110, file=filename,status='new'& 
                ,form='formatted', access='direct', recl=17,iostat=rc)
                if(rc/=0) print *,'open file failed!'

                write(filenumber_iteration,'(i5)')it
                filename = './output_inversion/forwavefield_vx'//trim(adjustl(filenumber_iteration))//'.dat'
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
            szz(izs,ixs)    = szz(izs,ixs) + source_term(it)
            sxx(izs,ixs)    = sxx(izs,ixs) + source_term(it)
            
        end do

        !********************************************************************
        ! 2. Forward wavefield modeling create direct wave only
        !********************************************************************


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
        
        do it=1,nt  !Loop over time
            ! if(mod(it,it_display) == 0) print *, it
        !   Compute stresses and particle velocities:
            call stepvelstress(npml,lambda_mig, mu_mig, rho_mig, nz, nx, aa, vz, vx, dz,  &
                dx, dt, szz, sxx, sxz, mem_dvz_dz, mem_dvz_dx, mem_dvx_dz,  &
                mem_dvx_dx, mem_dszz_dz, mem_dsxx_dx, mem_dsxz_dz,        &
                mem_dsxz_dx, a_z, b_z, K_z, a_z_half, b_z_half, K_z_half,  &
                a_x, b_x, K_x, a_x_half, b_x_half, K_x_half)

            if(mod(it,it_display) == -1)then
                write(filenumber_iteration,'(i5)')it
                filename = './output_inversion/wavefield_vz_Dwave_only'//trim(adjustl(filenumber_iteration))//'.dat'
                open(110, file=filename)
                close(110,status='delete')
                open(110, file=filename,status='new'& 
                ,form='formatted', access='direct', recl=17,iostat=rc)
                if(rc/=0) print *,'open file failed!'

                write(filenumber_iteration,'(i5)')it
                filename = './output_inversion/wavefield_vx_Dwave_only'//trim(adjustl(filenumber_iteration))//'.dat'
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

            !remove the direct wave from the seimogram data 
            vz_seismo(1:nx,it) = vz_seismo(1:nx,it)  - vz(izs,1:nx) 
            vx_seismo(1:nx,it) = vx_seismo(1:nx,it)  - vx(izs,1:nx) 

        !   Add source term to normal stresss sxx and szz:
            t = real(it-1)*dt

        !   First derivative of Gaussian:
            source_term(it) = -factor*2.*a*(t-t0)*exp(-a*(t-t0)**2)
            szz(izs,ixs)    = szz(izs,ixs) + source_term(it)
            sxx(izs,ixs)    = sxx(izs,ixs) + source_term(it)
        end do

        





        ! write(filenumber_iteration,'(i5)')ixs
        ! open(10, file='./output_inversion/trueseismogram_vz'//trim(adjustl(filenumber_iteration))//'.dat')
        ! close(10,status='delete')
        ! open(10, file='./output_inversion/trueseismogram_vz'//trim(adjustl(filenumber_iteration))//'.dat',status='new'& 
        ! ,form='formatted', access='direct', recl=17,iostat=rc)
        ! if(rc/=0) print *,'open file failed!'

        ! open(11, file='./output_inversion/trueseismogram_vx'//trim(adjustl(filenumber_iteration))//'.dat')
        ! close(11,status='delete')
        ! open(11, file='./output_inversion/trueseismogram_vx'//trim(adjustl(filenumber_iteration))//'.dat',status='new'& 
        ! ,form='formatted', access='direct', recl=17,iostat=rc)
        ! if(rc/=0) print *,'open file failed!'
        ! do i = 1,nx
        !     do j = 1,nt
        !         write(10,'(f17.8)',rec=j+(i-1)*nt) vz_seismo(i,j)
        !         write(11,'(f17.8)',rec=j+(i-1)*nt) vx_seismo(i,j)
        !     end do
        ! end do
        ! close(10)
        ! close(11)




        !********************************************************************
        ! 3. RTM
        !********************************************************************

        

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
            call stepvelstress(npml,lambda_init, mu_init, rho_init, nz, nx, aa, vz, vx, dz,  &
                dx, dt, szz, sxx, sxz, mem_dvz_dz, mem_dvz_dx, mem_dvx_dz,  &
                mem_dvx_dx, mem_dszz_dz, mem_dsxx_dx, mem_dsxz_dz,        &
                mem_dsxz_dx, a_z, b_z, K_z, a_z_half, b_z_half, K_z_half,  &
                a_x, b_x, K_x, a_x_half, b_x_half, K_x_half)

            

            if(mod(it,it_display) == -1)then
                write(filenumber_iteration,'(i5)')it
                filename = './output_inversion/backwavefield_vz'//trim(adjustl(filenumber_iteration))//'.dat'
                open(110, file=filename)
                close(110,status='delete')
                open(110, file=filename,status='new'& 
                ,form='formatted', access='direct', recl=17,iostat=rc)
                if(rc/=0) print *,'open file failed!'

                write(filenumber_iteration,'(i5)')it
                filename = './output_inversion/backwavefield_vx'//trim(adjustl(filenumber_iteration))//'.dat'
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
                image_vz = image_vz + vz_wavefield(1:nz,1:nx,nt/incre_t-itt+1)*vz(1:nz,1:nx)
                image_vx = image_vx + vx_wavefield(1:nz,1:nx,nt/incre_t-itt+1)*vx(1:nz,1:nx)
            end if

        !   backward propagate true seismogram:
            vz(izs,1:nx) = vz_seismo(1:nx,nt-it+1)
            vx(izs,1:nx) = vx_seismo(1:nx,nt-it+1)

        end do

        ! write(filenumber_iteration,'(i5)')ixs
        ! open(10, file='./output_inversion/image_vz'//trim(adjustl(filenumber_iteration))//'.dat')
        ! close(10,status='delete')
        ! open(10, file='./output_inversion/image_vz'//trim(adjustl(filenumber_iteration))//'.dat',status='new'& 
        ! ,form='formatted', access='direct', recl=17,iostat=rc)
        ! if(rc/=0) print *,'open file failed!'

        ! open(11, file='./output_inversion/image_vx'//trim(adjustl(filenumber_iteration))//'.dat')
        ! close(11,status='delete')
        ! open(11, file='./output_inversion/image_vx'//trim(adjustl(filenumber_iteration))//'.dat',status='new'& 
        ! ,form='formatted', access='direct', recl=17,iostat=rc)
        ! if(rc/=0) print *,'open file failed!'

        ! do i=1,nx
        ! do j=1,nz
        !         write(10,'(f17.8)',rec=j + (i-1)*nz) image_vz(j,i)
        !         write(11,'(f17.8)',rec=j + (i-1)*nz) image_vx(j,i)
        ! end do
        ! end do
        ! close(10)
        ! close(11)

    end do


    write(filenumber_iteration,'(i5)')shot_num
    open(10, file='./output_inversion/Fortran_rtm'//trim(adjustl(filenumber_iteration))//'.dat')
    close(10,status='delete')
    open(10, file='./output_inversion/Fortran_rtm'//trim(adjustl(filenumber_iteration))//'.dat',status='new'& 
    ,form='formatted', access='direct', recl=17,iostat=rc)
    if(rc/=0) print *,'open file failed!'

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 0*nz*nx) image_vz(j,i)/100
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 1*nz*nx) image_vx(j,i)/100
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 2*nz*nx) vp_init(j,i)
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 3*nz*nx) vs_init(j,i)
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 4*nz*nx) rho_init(j,i)
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 5*nz*nx) vp(j,i)
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 6*nz*nx) vs(j,i)
    end do
    end do

    do i=1,nx
    do j=1,nz
            write(10,'(f17.8)',rec=j + (i-1)*nz + 7*nz*nx) rho(j,i)
    end do
    end do

    close(10)
end do

! call exit
end program migration

include "./common/stepvelstress.f90"
include "./common/pml.f90"
!********************************************************************
! Main Program End
!********************************************************************
