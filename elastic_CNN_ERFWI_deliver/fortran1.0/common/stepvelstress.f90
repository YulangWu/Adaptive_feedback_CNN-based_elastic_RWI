!! SEISMIC_CPML Version 1.1.1, November 2009.
!
! Copyright Universite de Pau et des Pays de l'Adour, 
! CNRS and INRIA, France.
! Contributors: 
!       Dimitri Komatitsch, dimitri DOT komatitsch aT univ-pau DOT fr
!       and Roland Martin, roland DOT martin aT univ-pau DOT fr
!
! This software is a computer program whose purpose is to solve the 
! two-dimensional isotropic elastic wave equation using a 
! finite-difference method with Convolutional Perfectly Matched 
! Layer (C-PML) conditions.
!
! This software is governed by the CeCILL license under French law and
! abiding by the rules of distribution of free software. You can use,
! modify and/or redistribute the software under the terms of the CeCILL
! license as circulated by CEA, CNRS and INRIA at the following URL
! "http://www.cecill.info".
!
! As a counterpart to the access to the source code and rights to copy,
! modify and redistribute granted by the license, users are provided
! only with a limited warranty and the software's author, the holder 
! of the economic rights, and the successive licensors have only 
! limited liability.
!
! In this respect, the user's attention is drawn to the risks 
! associated with loading, using, modifying and/or developing or 
! reproducing the software by the user in light of its specific status 
! of free software, that may mean that it is complicated to manipulate, 
! and that also therefore means that it is reserved for developers and 
! experienced professionals having in-depth computer knowledge. Users 
! are therefore encouraged to load and test the software's suitability 
! as regards their requirements in conditions enabling the security of 
! their systems and/or data to be ensured and, more generally, to use 
! and operate it in the same conditions as regards security.
!
! The full text of the license is available at the end of this program
! and in file "LICENSE".
!-----------------------------------------------------------------------
! 2D elastic finite-difference code in velocity and stress formulation
! with Convolutional-PML (C-PML) absorbing conditions for an isotropic
! medium

! Dimitri Komatitsch, University of Pau, France, April 2007.
! Fourth-order implementation by Dimitri Komatitsch and Roland Martin,
! University of Pau, France, August 2007.
!
! The staggered-grid formulation of Madariaga (1976) and Virieux (1986)
! is used:
!
!            ^ y
!            |
!            |
!
!            +-------------------+
!            |                   |
!            |                   |
!            |                   |
!            |                   |
!            |        v_y        |
!   sigma_xy +---------+         |
!            |         |         |
!            |         |         |
!            |         |         |
!            |         |         |
!            |         |         |
!            +---------+---------+  ---> x
!           v_x    sigma_xx
!                  sigma_yy
!
! but a fourth-order spatial operator is used instead of a second-order
! operator
! as in program seismic_CPML_2D_iso_second.f90 . You can type the
! following command
! to see the changes that have been made to switch from the second-order
! operator
! to the fourth-order operator:
!
! diff seismic_CPML_2D_isotropic_second_order.f90
! seismic_CPML_2D_isotropic_fourth_order.f90

! The C-PML implementation is based in part on formulas given in Roden
! and Gedney (2000)
!
! If you use this code for your own research, please cite some (or all)
! of these articles:
!
! @ARTICLE{MaKoEz08,
! author = {Roland Martin and Dimitri Komatitsch and Abdelaaziz
! Ezziani},
! title = {An unsplit convolutional perfectly matched layer improved at
! grazing
!          incidence for seismic wave equation in poroelastic media},
! journal = {Geophysics},
! year = {2008},
! volume = {73},
! pages = {T51-T61},
! number = {4},
! doi = {10.1190/1.2939484}}
!
! @ARTICLE{MaKoGe08,
! author = {Roland Martin and Dimitri Komatitsch and Stephen D. Gedney},
! title = {A variational formulation of a stabilized unsplit
! convolutional perfectly
!          matched layer for the isotropic or anisotropic seismic wave
!          equation},
! journal = {Computer Modeling in Engineering and Sciences},
! year = {2008},
! volume = {37},
! pages = {274-304},
! number = {3}}
!
! @ARTICLE{RoGe00,
! author = {J. A. Roden and S. D. Gedney},
! title = {Convolution {PML} ({CPML}): {A}n Efficient {FDTD}
! Implementation
!          of the {CFS}-{PML} for Arbitrary Media},
! journal = {Microwave and Optical Technology Letters},
! year = {2000},
! volume = {27},
! number = {5},
! pages = {334-339},
! doi = {10.1002/1098-2760(20001205)27:5<334::AID-MOP14>3.0.CO;2-A}}
!
! @ARTICLE{KoMa07,
! author = {Dimitri Komatitsch and Roland Martin},
! title = {An unsplit convolutional {P}erfectly {M}atched {L}ayer
! improved
!          at grazing incidence for the seismic wave equation},
! journal = {Geophysics},
! year = {2007},
! volume = {72},
! number = {5},
! pages = {SM155-SM167},
! doi = {10.1190/1.2757586}}
!
! To display the 2D results as color images, use:
!
!   " display image*.gif " or " gimp image*.gif "
!
! or
!
!   " montage -geometry +0+3 -rotate 90 -tile 1x21 image*Vx*.gif
!   allfiles_Vx.gif "
!   " montage -geometry +0+3 -rotate 90 -tile 1x21 image*Vy*.gif
!   allfiles_Vy.gif "
!   then " display allfiles_Vx.gif " or " gimp allfiles_Vx.gif "
!   then " display allfiles_Vy.gif " or " gimp allfiles_Vy.gif "
!

! IMPORTANT : all our CPML codes work fine in single precision as well
! (which is significantly faster).
!             If you want you can thus force automatic conversion to
!             single precision at compile time
!             or change all the declarations and constants in the code
!             from double precision to single.
!-----------------------------------------------------------------------
   SUBROUTINE stepvelstress(npml,lambda, mu, rho, nz, nx, aa, vz, vx, &
       dz, dx, dt, szz, sxx, sxz, memory_dvz_dz, &
       memory_dvz_dx, memory_dvx_dz, memory_dvx_dx, &
       memory_dszz_dz, memory_dsxx_dx, memory_dsxz_dz, &
       memory_dsxz_dx, a_z, b_z, K_z, a_z_half, b_z_half, &
       K_z_half, a_x, b_x, K_x, a_x_half, b_x_half, K_x_half)
!
        implicit none
		integer,intent(in)::npml
        integer nz, nx, iz, ix
        real(kind=4), dimension(-2:nz+3,-2:nx+3) :: vp,vs,rho,vz,vx,lambda,mu
        real(kind=4)  aa(4), dz, dx, dt

        !8th-order:
        real(kind=4), dimension(-2:nz+3,-2:nx+3) :: szz,sxx,sxz

        ! to interpolate material parameters at the right location in 
        ! the staggered grid cell
        real(kind=4) lambda_half_z, mu_half_z, lambda_plus_two_mu_half_z
        real(kind=4) mu_half_x, rho_half_z_half_x

        ! arrays for the memory variables could declare these arrays 
        !in PML only to save a lot of memory, but proof of concept only here
        real(kind=4), dimension(-2:nz+3,-2:nx+3) :: memory_dvz_dz, &
       memory_dvz_dx, memory_dvx_dz, memory_dvx_dx, memory_dszz_dz,&
       memory_dsxx_dx, memory_dsxz_dz, memory_dsxz_dx

        real(kind=4) :: value_dvz_dz, value_dvz_dx, value_dvx_dz, value_dvx_dx
        real(kind=4) :: value_dszz_dz, value_dsxx_dx, &
                 value_dsxz_dz, value_dsxz_dx

        ! 1D arrays for the damping profiles
        real(kind=4), dimension(nz) :: a_z, b_z, K_z
        real(kind=4), dimension(nz) :: a_z_half, b_z_half, K_z_half
        real(kind=4), dimension(nx) :: a_x, b_x, K_x
        real(kind=4), dimension(nx) :: a_x_half, b_x_half, K_x_half

           !------------------------------------------------------------
           ! compute stress s and update memory variables for C-PML
           !------------------------------------------------------------

           do ix = 2,nx
              do iz = 1,nz-1
                 lambda_half_z = 0.5e0*(lambda(iz+1,ix) + lambda(iz,ix))
                 mu_half_z = 0.5e0 * (mu(iz+1,ix) + mu(iz,ix))
                 lambda_plus_two_mu_half_z=lambda_half_z+2.e0*mu_half_z

                 value_dvz_dz = (aa(1)*vz(iz+1,ix) - aa(1)*vz(iz,ix) &
                             - aa(2)*vz(iz+2,ix)  + aa(2)*vz(iz-1,ix) &
                             + aa(3)*vz(iz+3,ix)  - aa(3)*vz(iz-2,ix) &
                             - aa(4)*vz(iz+4,ix)  + aa(4)*vz(iz-3,ix)) &
                             / dz

                 value_dvx_dx = (aa(1)*vx(iz,ix)  - aa(1)*vx(iz,ix-1) &
                             - aa(2)*vx(iz,ix+1) + aa(2)*vx(iz,ix-2) &
                             + aa(3)*vx(iz,ix+2) - aa(3)*vx(iz,ix-3) &
                             - aa(4)*vx(iz,ix+3) + aa(4)*vx(iz,ix-4)) &
                             / dx

                 memory_dvz_dz(iz,ix) = b_z_half(iz) &
                * memory_dvz_dz(iz,ix) + a_z_half(iz) * value_dvz_dz

                 memory_dvx_dx(iz,ix) = b_x(ix) &
                * memory_dvx_dx(iz,ix) + a_x(ix) * value_dvx_dx

                 value_dvz_dz = value_dvz_dz / K_z_half(iz) &
                             + memory_dvz_dz(iz,ix)
                 value_dvx_dx = value_dvx_dx / K_x(ix) &
                             + memory_dvx_dx(iz,ix)

                 szz(iz,ix) = szz(iz,ix) &
                + (lambda_plus_two_mu_half_z * value_dvz_dz &
                + lambda_half_z * value_dvx_dx) * dt

                 sxx(iz,ix) = sxx(iz,ix) &
                + (lambda_half_z * value_dvz_dz &
                + lambda_plus_two_mu_half_z * value_dvx_dx) * dt
              enddo
           enddo

           do ix = 1,nx-1
              do iz = 2,nz
                 mu_half_x = 0.5e0 * (mu(iz,ix+1) + mu(iz,ix))

                 value_dvx_dz = (aa(1)*vx(iz,ix)  - aa(1)*vx(iz-1,ix) &
                             - aa(2)*vx(iz+1,ix) + aa(2)*vx(iz-2,ix) &
                             + aa(3)*vx(iz+2,ix) - aa(3)*vx(iz-3,ix) &
                             - aa(4)*vx(iz+3,ix) + aa(4)*vx(iz-4,ix))  &
                           / dz

                 value_dvz_dx = (aa(1)*vz(iz,ix+1) - aa(1)*vz(iz,ix) &
                             - aa(2)*vz(iz,ix+2)  + aa(2)*vz(iz,ix-1) &
                             + aa(3)*vz(iz,ix+3)  - aa(3)*vz(iz,ix-2) &
                             - aa(4)*vz(iz,ix+4)  + aa(4)*vz(iz,ix-3)) &
                             / dx

                 memory_dvx_dz(iz,ix) = b_z(iz) &
                * memory_dvx_dz(iz,ix) + a_z(iz) * value_dvx_dz

                 memory_dvz_dx(iz,ix) = b_x_half(ix) &
                * memory_dvz_dx(iz,ix) + a_x_half(ix) * value_dvz_dx

                 value_dvx_dz = value_dvx_dz / K_z(iz) &
                             + memory_dvx_dz(iz,ix)
                 value_dvz_dx = value_dvz_dx / K_x(ix) &
                             + memory_dvz_dx(iz,ix)

                 sxz(iz,ix) = sxz(iz,ix) &
                + mu_half_x * (value_dvx_dz + value_dvz_dx) * dt
              enddo
           enddo

           !--------------------------------------------------------
           ! compute velocity and update memory variables for C-PML
           !--------------------------------------------------------


           do ix = 2,nx
              do iz = 2,nz
                 value_dszz_dz = &
                (aa(1)*szz(iz,ix)    - aa(1)*szz(iz-1,ix) &
                - aa(2)*szz(iz+1,ix) + aa(2)*szz(iz-2,ix) &
                + aa(3)*szz(iz+2,ix) - aa(3)*szz(iz-3,ix) &
                - aa(4)*szz(iz+3,ix) + aa(4)*szz(iz-4,ix)) / dz

                 value_dsxz_dx = &
                (aa(1)*sxz(iz,ix)    - aa(1)*sxz(iz,ix-1) &
                - aa(2)*sxz(iz,ix+1) + aa(2)*sxz(iz,ix-2) &
                + aa(3)*sxz(iz,ix+2) - aa(3)*sxz(iz,ix-3) &
                - aa(4)*sxz(iz,ix+3) + aa(4)*sxz(iz,ix-4)) / dx

                 memory_dszz_dz(iz,ix) = b_z(iz) &
                * memory_dszz_dz(iz,ix) + a_z(iz)*value_dszz_dz 
                 memory_dsxz_dx(iz,ix) = b_x(ix) &
                * memory_dsxz_dx(iz,ix) + a_x(ix)*value_dsxz_dx

                 value_dszz_dz = value_dszz_dz / K_z(iz) &
                                  + memory_dszz_dz(iz,ix)
                 value_dsxz_dx = value_dsxz_dx / K_x(ix) &
                                  + memory_dsxz_dx(iz,ix)

                 vz(iz,ix) = vz(iz,ix) & 
                 + (value_dszz_dz+value_dsxz_dx)* dt /rho(iz,ix)
              enddo
           enddo

           do ix = 1,nx-1
              do iz = 1,nz-1
                 rho_half_z_half_x = 0.25e0 * (rho(iz,ix) &
                + rho(iz+1,ix) + rho(iz+1,ix+1) + rho(iz,ix+1))

                 value_dsxz_dz = & 
                (aa(1)*sxz(iz+1,ix)  - aa(1)*sxz(iz,ix) &
                - aa(2)*sxz(iz+2,ix) + aa(2)*sxz(iz-1,ix) &
                + aa(3)*sxz(iz+3,ix) - aa(3)*sxz(iz-2,ix) &
                - aa(4)*sxz(iz+4,ix) + aa(4)*sxz(iz-3,ix)) / dz

                 value_dsxx_dx = &
                (aa(1)*sxx(iz,ix+1)  - aa(1)*sxx(iz,ix) &
                - aa(2)*sxx(iz,ix+2) + aa(2)*sxx(iz,ix-1) &
                + aa(3)*sxx(iz,ix+3) - aa(3)*sxx(iz,ix-2) &
                - aa(4)*sxx(iz,ix+4) + aa(4)*sxx(iz,ix-3)) / dx

                 memory_dsxz_dz(iz,ix) = &
                b_z_half(iz) * memory_dsxz_dz(iz,ix) &
                + a_z_half(iz) * value_dsxz_dz

                 memory_dsxx_dx(iz,ix) = &
                b_x_half(ix) * memory_dsxx_dx(iz,ix) &
                + a_x_half(ix) * value_dsxx_dx

                 value_dsxz_dz = value_dsxz_dz / K_z_half(iz) &
                + memory_dsxz_dz(iz,ix)

                 value_dsxx_dx = value_dsxx_dx / K_x_half(ix) &
                + memory_dsxx_dx(iz,ix)

                 vx(iz,ix) = vx(iz,ix) &
                + (value_dsxz_dz + value_dsxx_dx) &
                * dt / rho_half_z_half_x
              enddo
           enddo

        RETURN
    END SUBROUTINE stepvelstress    