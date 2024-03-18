#!/bin/csh -f

ifort -O2 migration1.f90 -o migration1.exe 
ifort -O2 migration2.f90 -o migration2.exe 
ifort -O2 migration3.f90 -o migration3.exe 
ifort -O2 migration4.f90 -o migration4.exe 


nohup ./migration1.exe &
nohup ./migration2.exe &
nohup ./migration3.exe &
nohup ./migration4.exe &



