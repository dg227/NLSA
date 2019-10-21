function a = getProjectedData( obj, varargin )
% GETDPROJECTEDDATA Get projected target data onto the diffusion eigenfunctions of 
% an nlsaModel object
%
% Modified 2014/06/24

a = getProjectedData( getPrjComponent( obj ), varargin{ : } );

