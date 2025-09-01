function phi = getSclIsrDiffusionEigenfunctions( obj, varargin )
% GETSCLISRDIFFUSIONEIGENFUNCTIONS Get scaled ISR eigenfunctions of an 
% nlsaModel_scl object
%
% Modified 2014/07/28

phi = getEigenfunctions( getSclIsrDiffusionOperator( obj ), varargin{ : } );
