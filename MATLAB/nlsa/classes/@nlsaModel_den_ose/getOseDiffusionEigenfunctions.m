function phi = getOseDiffusionEigenfunctions( obj, varargin )
% GETOSEDIFFUSIONEIGENFUNCTIONS Get OSE diffusion eigenfunctions of an 
% nlsaModel_den_ose object
%
% Modified 2018/07/01

phi = getEigenfunctions( getOseDiffusionOperator( obj ), varargin{ : } );
