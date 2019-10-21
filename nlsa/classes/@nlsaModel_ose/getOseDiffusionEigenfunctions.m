function phi = getOseDiffusionEigenfunctions( obj, varargin )
% GETOSEDIFFUSIONEIGENFUNCTIONS Get OSE diffusion eigenfunctions of an 
% nlsaModel_ose object
%
% Modified 2014/02/22

phi = getEigenfunctions( getOseDiffusionOperator( obj ), varargin{ : } );
