#!/bin/csh -f

ifort -O2 inversion_data.f90 -o inversion_data.exe 
./inversion_data.exe




