%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Simulation files
Miguel Parada Contzen
Conce, May 2024
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

The files contained in this folder are for computing the examples in the paper

	Parada Contzen, Miguel. 'Distributed voltage regulation and state of charge consensus control of autonomous battery energy storage systems in medium voltage circuits.', 2024.

All the simulation figures in the paper are obtained using conventional desktop hardware and the following Python 3.10 scripts:

	- example_paper.py: Main file for closed and open loop simulation of described system and power control algorithm. It produces all simulation figures in the paper.
	- circuit_fun.py: Functions used for defining arbitrary circuits, in terms of topology and parameters values.
	- loads_lib.py: Functions for defining arbitrary load profiles based on domiciliary behavior including different load devices and the charging of electric vehicles
	- time_fun.py: Functions for interpoling the dynamic behavior of photo-voltaic, batteries, and load profiles.
	- NR_fun.py: Functions for running Newton-Raphson algorithms for power flow calculations.