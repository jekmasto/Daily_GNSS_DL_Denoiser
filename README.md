# Daily_GNSS_DL_Denoiser
Models and code underlying the paper:

Denoising Daily Displacement GNSS-Time series using Deep Neural Networks In a Near Real-Time Framing

by Mastella G.1, Bedford J.2,  Corbi F.3, Funiciello F.1

1 Università “Roma TRE,”, Rome, Italy, Dip Scienze, Laboratory of Experimental Tectonics.
2 Institut für Geologie, Mineralogie und Geophysik, Ruhr-Universität Bochum, Bochum, 44801, Germany.
3 Istituto di Geologia Ambientale e Geoingegneria – CNR c/o Dipartimento di Scienze della Terra, Sapienza Università di Roma, Rome, Italy.

*Corresponding author: Giacomo Mastella (giacomo.mastella@uniroma3.it)

Author of the Code: Giacomo Mastella and Jonathan Bedford
Contact  Giacomo Mastella (giacomo.mastella@uniroma3.it) if you have any problems using this code.

Make sure you have the following packages:

 - numpy  (I am using 1.23.4)
 - tensorflow2  (I am using 2.9.1)
 - scipy (1.3.3)
 - pandas
 - datetime

Additionaly the GrAtSiD code has to be installed. 
GrAtSiD, the Greedy Automatic Signal Decomposition algorithm is a sequential greedy linear regression algorithm that perform the trajectory modelling of GNSS time-series.
The code is avaliable from:
https://github.com/TectonicGeodesy-RUB/Gratsid
to cite:
Bedford, J. and Bevis, M., 2018. Greedy automatic signal decomposition and its application to daily GPS time series. Journal of Geophysical Research: Solid Earth, 123(8), pp.6992-7003.

