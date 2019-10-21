function phi = getIsrDiffusionEigenfunctions( obj, varargin )
% GETISRDIFFUSIONEIGENFUNCTIONS Get in-sample restriction diffusion 
% eigenfunctions of an nlsaModel_err object
%
% Modified 2014/07/27

phi = getEigenfunctions( getIsrDiffusionOperator( obj ), varargin{ : } );
