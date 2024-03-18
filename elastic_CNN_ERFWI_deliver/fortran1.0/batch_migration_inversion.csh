#!/bin/csh -f

ifort -O2 migration_inversion.f90 -o migration_inversion.exe 
./migration_inversion.exe




