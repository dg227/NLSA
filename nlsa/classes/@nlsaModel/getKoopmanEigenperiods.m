function T = getKoopmanEigenperiods( obj )
% GETKOOPMANEIGENVALUES Get Koopman eigenperiods of an nlsaModel object
%
% Modified 2020/06/16

T = getEigenperiods( getKoopmanOperator( obj ) );

