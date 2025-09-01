function T = getEigenperiods( obj )
% GETEIGENPERIODS  Get eigenperiods of an nlsaKoopmanOperator object
%
% Modified 2020/04/12

T = 2 * pi ./ getEigenfrequencies( obj );

