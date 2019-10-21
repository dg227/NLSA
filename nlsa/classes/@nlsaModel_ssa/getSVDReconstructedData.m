function x = getSVDReconstructedData( obj, varargin )
% GETRECONSTRUCTEDDATA Get SVD reconstructed target data of an nlsaModel_ssa 
% object
%
% Modified 2016/05/31

x = getData( getSVDReComponent( obj ), varargin{ : } );

