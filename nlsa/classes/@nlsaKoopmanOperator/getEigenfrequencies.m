function omega = getEigenfrequencies( obj )
% GETEIGENFREQUENCIES  Get eigenfrequencies of an nlsaKoopmanOperator object
%
% Modified 2020/04/12

omega = imag( getEigenvalues( obj ) );

