function omega = getKoopmanEigenfrequencies( obj )
% GETKOOPMANEIGENFREQUENCIES Get Koopman eigenfrequencies of an nlsaModel 
% object
%
% Modified 2020/08/07

omega = getEigenfrequencies( getKoopmanOperator( obj ) );
