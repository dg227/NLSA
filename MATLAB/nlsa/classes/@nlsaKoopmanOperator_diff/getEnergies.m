function E = getEnergies( obj )
% GETENERGIES  Read Dirichlet energies of an nlsaKoopmanOperator_diff object
%
% Modified 2020/04/11

load( fullfile( getEigenfunctionPath( obj ), getEigenvalueFile( obj ) ), ...
      'E' )
