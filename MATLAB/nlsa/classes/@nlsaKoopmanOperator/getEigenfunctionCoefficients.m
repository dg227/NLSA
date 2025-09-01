function c = getEigenfunctionCoefficients( obj )
% GETEIGENFUNCTIONCOEFFICIENTS  Read eigenfunction coefficients of an 
% nlsaKoopmanOperator object
%
% Modified 2020/04/11

load( fullfile( getEigenfunctionPath( obj ), ...
                getEigenfunctionCoefficientFile( obj ) ), 'c' )
