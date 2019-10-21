function x = getSVDReconstructedData( obj, varargin )
% GETRECONSTRUCTEDDATA Get SVD reconstructed target data of an nlsaModel object
%
% Modified 2016/01/24

x = getData( getSVDReComponent( obj ), varargin{ : } );

