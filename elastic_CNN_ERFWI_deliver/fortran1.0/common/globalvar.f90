!----------------------------------------------------------------
! Define common variables
!----------------------------------------------------------------
! Constant global parameter:
real(kind=4), parameter :: pi    = 4.d0*atan(1.d0) !constant value of pi
integer, parameter      :: int0  = 0    !constant integer 0
real(kind=4), parameter :: real0 = 0.d0 !constant real 0.d0

!----------------------------------------------------------------
! Vector component particle velocities:
real(kind=4), allocatable, dimension(:,:) :: vz, vx
! Normal and tangential stresses:
real(kind=4), allocatable, dimension(:,:) :: szz, sxx, sxz
!----------------------------------------------------------------
! To interpolate material parameters at the right location 
! in the staggered grid cell:
real(kind=4) lambda_half_z, mu_half_z, lambda_plus_two_mu_half_z
real(kind=4) mu_half_x
!----------------------------------------------------------------
! Wavefield temporal memory variables to solve elastodynamic equations
real(kind=4), allocatable,dimension (:,:) :: mem_dvz_dz, mem_dvz_dx
real(kind=4), allocatable,dimension (:,:) :: mem_dvx_dz, mem_dvx_dx, mem_dszz_dz
real(kind=4), allocatable,dimension (:,:) :: mem_dsxx_dx, mem_dsxz_dz,mem_dsxz_dx
!----------------------------------------------------------------
! Partial derivative variables:
real(kind=4) :: dvz_dz, dvz_dx, dvx_dz, dvx_dx, dszz_dz, dsxx_dx,dsxz_dz, dsxz_dx
!----------------------------------------------------------------
! 1D arrays for the damping profiles:
real(kind=4), allocatable, dimension (:) :: K_z,a_z,b_z,K_z_half,a_z_half, b_z_half 
real(kind=4), allocatable, dimension (:) :: K_x,a_x,b_x,K_x_half,a_x_half, b_x_half  
  
real(kind=4), allocatable, dimension (:) :: alpha_z_half,alpha_z,d_z_half,d_z 
real(kind=4), allocatable, dimension (:) :: alpha_x_half,alpha_x,d_x_half ,d_x
!----------------------------------------------------------------
! Source function:
real(kind=4) :: a, t
real(kind=4), allocatable :: source_term(:)
!----------------------------------------------------------------
! Other variables:
integer      :: iz, ix, it  ! Counter variables
real(kind=4)  :: CourantNum  ! Courant number in the Courant-Friedrichs-Lewy (CFL)  
                             ! condition should be less than 0.6 for elastic stability
real(kind=4)  :: aa(4)       ! FD coefficient array
!
! Arrays for elastic model properties:
real(kind=4), allocatable, dimension(:,:) :: vp, vs, rho, lambda, mu

!----------------------------------------------------------------
! Arrays for initial value reconstructions:
real(kind=4), dimension(:,:,:), allocatable :: vz_wavefield,vx_wavefield
real(kind=4), dimension(:,:,:), allocatable :: szz_wavefield
real(kind=4), dimension(:,:,:), allocatable :: sxx_wavefield
real(kind=4), dimension(:,:,:), allocatable :: sxz_wavefield
!----------------------------------------------------------------
! Variables:
integer     :: npml        ! number of PML points
integer     :: nx          ! number of horizontal (x) grid lines including 
                            ! the npml grid lines
real(kind=4) :: dx          ! the increments (meters) in the x-directions
integer     :: nz          ! number of vertical (z) grid lines including 
                            ! the npml grid lines
real(kind=4) :: dz          ! the increments (meters) in the z-directions
real(kind=4) :: f0          ! dominant frequency (Hertz)
real(kind=4) :: t0          ! the time at which the amplitude of the first 
                            ! derivative of the Gaussian wavelet (the slope  
                            ! of the Gaussian wavelet) is zero and the  
                            ! amplitude of the Gaussian wavelet is maximum
real(kind=4) :: factor      ! a scalar to multiply the source amplitude by
integer     :: izs,ixs     ! the (z,x) computational mesh grid location 
                            ! of the source (in units of grid points), 
                            ! including CPML points
integer     :: nt          ! number of time steps to simulate
real(kind=4) :: dt,temp          ! the temporal sampling (seconds)
integer     :: it_display  ! the increment of the time steps to display

!----------------------------------------------------------------
! Temporary variables:
! The file name for input and output:
character(len=100)::filename
character(len=100)::filenumber_frequency
character(len=100)::filenumber_iteration
character(len=100)::filenumber_repetition
integer            :: i,j,rc
character(len=100)::truevp
character(len=100)::initvp
!----------------------------------------------------------------
! for RTM only
integer::incre_t = 1,itt
integer::incre_s = 20
integer::shot_num,num_shot = 4
real(kind=4), allocatable, dimension(:,:) :: image_vz,image_vx
real(kind=4), allocatable, dimension(:,:) :: vp_init, vs_init, rho_init, lambda_init, mu_init
real(kind=4), allocatable, dimension(:,:) :: vp_mig, vs_mig, rho_mig, lambda_mig, mu_mig
real(kind=4), dimension(:,:), allocatable :: vz_seismo,vx_seismo

!
! Input the parameters:
!
open(1,file='./parameter.txt',status='old'&
    ,form='formatted',iostat=rc)

read(1,*)npml
read(1,*)nx
read(1,*)dx
read(1,*)nz
read(1,*)dz
read(1,*)f0
read(1,*)t0
read(1,*)factor
read(1,*)izs
read(1,*)ixs
read(1,*)nt
read(1,*)dt
read(1,*)it_display
read(1,*)truevp
read(1,*)initvp
read(1,*)incre_t
read(1,*)incre_s
read(1,*)num_shot
close(1)
!
! Display the input parameters:
!
print *,"npml = ",npml
print *,"nx = ",nx
print *,"dx = ",dx
print *,"nz = ",nz
print *,"dz = ",dz
print *,"f0 = ",f0
print *,"t0 = ",t0
print *,"factor = ",factor
print *,"izs = ",izs
print *,"ixs = ",ixs
print *,"nt = ",nt
print *,"dt = ",dt
print *,"it_display = ",it_display
print *,"truevp = ",truevp
print *,"initvp = ",initvp
print *,"incre_t = ",incre_t
print *,"incre_s = ",incre_s
print *,"num_shot = ",num_shot
a = pi*pi*f0*f0

! ================================================================
! Holberg (1987) coefficients, taken from
! @ARTICLE{Hol87,
! author  = {O. Holberg},
! title   = {Computational aspects of the choice of operator and  
!            sampling interval for numerical differentiation in   
!            large-scale simulation of wave phenomena},
! journal = {Geophysical Prospecting},
! year    = {1987},
! volume  = {35},
! pages   = {629-655}}
aa(1)     = 1.231666        ! 8th-order FD weighting coefficients
aa(2)     = 0.1041182
aa(3)     = 0.0263707       
aa(4)     = 0.003570998
! ----------------------------------------------------------------
! Allocate space for dynamic arrays:
allocate(vz(-2:nz+3,-2:nx+3),vx(-2:nz+3,-2:nx+3), &
    szz(-2:nz+3,-2:nx+3),sxx(-2:nz+3,-2:nx+3),sxz(-2:nz+3,-2:nx+3), &
    lambda(-2:nz+3,-2:nx+3),mu(-2:nz+3,-2:nx+3)) 
allocate(mem_dvz_dz(-2:nz+3,-2:nx+3), &
    mem_dvz_dx(-2:nz+3,-2:nx+3), mem_dvx_dz(-2:nz+3,-2:nx+3), &
    mem_dvx_dx(-2:nz+3,-2:nx+3), mem_dszz_dz(-2:nz+3,-2:nx+3), &
    mem_dsxx_dx(-2:nz+3,-2:nx+3), mem_dsxz_dz(nz+6,nx+6), &
    mem_dsxz_dx(-2:nz+3,-2:nx+3))
allocate(d_z(nz), K_z(nz), alpha_z(nz), a_z(nz), b_z(nz), &
    d_z_half(nz), K_z_half(nz), alpha_z_half(nz), a_z_half(nz), &
    b_z_half(nz))
allocate(d_x(nx), K_x(nx), alpha_x(nx), a_x(nx), b_x(nx), &
    d_x_half(nx), K_x_half(nx), alpha_x_half(nx), a_x_half(nx), &
    b_x_half(nx))
allocate(source_term(nt))
allocate(vp(-2:nz+3,-2:nx+3), vs(-2:nz+3,-2:nx+3), &
    rho(-2:nz+3,-2:nx+3))

! ----------------------------------------------------------------
! for RTM only
allocate(vz_wavefield(1:nz,1:nx,nt/incre_t), vx_wavefield(1:nz,1:nx,nt/incre_t))
allocate(image_vz(1:nz,1:nx),image_vx(1:nz,1:nx))
allocate(vp_init(-2:nz+3,-2:nx+3), vs_init(-2:nz+3,-2:nx+3), &
    rho_init(-2:nz+3,-2:nx+3),lambda_init(-2:nz+3,-2:nx+3), mu_init(-2:nz+3,-2:nx+3))
    allocate(vp_mig(-2:nz+3,-2:nx+3), vs_mig(-2:nz+3,-2:nx+3), &
    rho_mig(-2:nz+3,-2:nx+3),lambda_mig(-2:nz+3,-2:nx+3), mu_mig(-2:nz+3,-2:nx+3))
allocate(vz_seismo(1:nx,nt), vx_seismo(1:nx,nt))