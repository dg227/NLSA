function path = getEigenfunctionPath( obj )
% GETOPERATORPATH  Get eigenfunction path of an nlsaKoopmanOperator 
% object
%
% Modified 2020/04/08

path = fullfile( getPath( obj ), getEigenfunctionSubpath( obj ) );
