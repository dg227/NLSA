function gamma = getEigenvalues( obj )
% GETEIGENVALUES  Read eigenvalues of an nlsaKoopmanOperator object
%
% Modified 2020/04/11

load( fullfile( getEigenfunctionPath( obj ), getEigenvalueFile( obj ) ), ...
      'gamma' )
