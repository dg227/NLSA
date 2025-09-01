function c = getLeftEigenfunctionCoefficients( obj )
% GETLEFTEIGENFUNCTIONCOEFFICIENTS  Read left eigenfunction coefficients of an 
% nlsaKoopmanOperator object
%
% Modified 2020/08/27

load( fullfile( getEigenfunctionPath( obj ), ...
                getLeftEigenfunctionCoefficientFile( obj ) ), 'c' )
