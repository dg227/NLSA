function x = getReconstructedData( obj, varargin )
% GETRECONSTRUCTEDDATA Get reconstructed target data of an nlsaModel_ssa object
%
% Modified 2016/05/31

x = getData( getRecComponent( obj ), varargin{ : } );

