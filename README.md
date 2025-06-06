# Daily_GNSS_DL_Denoiser
Models and code underlying the paper:

Denoising Daily Displacement GNSS Time series using Deep Neural Networks In a Near Real-Time Framing: a Single-Station Method

by Mastella G.1, Bedford J.2,  Corbi F.3, Funiciello F.1

1 Università “Sapienza”, Departimento di Scienze della Terra, Rome, Italy.
2 Institut für Geologie, Mineralogie und Geophysik, Ruhr-Universität Bochum, Bochum, 44801, Germany.
3 Istituto di Geologia Ambientale e Geoingegneria – CNR c/o Dipartimento di Scienze della Terra, Sapienza Università di Roma, Rome, Italy.

*Corresponding author: Giacomo Mastella (giacomo.mastella@uniroma1.it)

Author of the Code: Giacomo Mastella and Jonathan Bedford.

Contact  Giacomo Mastella (giacomo.mastella@uniroma1.it) if you have any problems using this code.

Make sure you have the following packages:

 - numpy  (I am using 1.23.4)
 - tensorflow2  (I am using 2.9.1)
 - scipy (1.3.3)
 - pandas
 - datetime

Additionaly the GrAtSiD code has to be installed. 

GrAtSiD, the Greedy Automatic Signal Decomposition algorithm is a sequential greedy linear regression algorithm that performs the trajectory modelling of GNSS time-series.

The code is avaliable from:
https://github.com/TectonicGeodesy-RUB/Gratsid

to cite:
Bedford, J. and Bevis, M., 2018. Greedy automatic signal decomposition and its application to daily GPS time series. Journal of Geophysical Research: Solid Earth, 123(8), pp.6992-7003.

Models will be avaliable soon on a zenodo repository.

if you want to use the GAN for generating synthetics residuals spectrograms, also the librosa package has to be installed
The code is avaliable from:
https://librosa.org/doc/latest/index.html

to cite:
McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.
https://zenodo.org/record/7746972,

