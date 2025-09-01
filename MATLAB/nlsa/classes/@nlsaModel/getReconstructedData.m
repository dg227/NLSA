function x = getReconstructedData( obj, varargin )
% GETRECONSTRUCTEDDATA Get reconstructed target data of an nlsaModel object
%
% Modified 2015/12/14

x = getData( getRecComponent( obj ), varargin{ : } );

