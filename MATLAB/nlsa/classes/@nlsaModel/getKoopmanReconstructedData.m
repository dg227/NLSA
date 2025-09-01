function x = getKoopmanReconstructedData( obj, varargin )
% GETKOOPMANRECONSTRUCTEDDATA Get Koopman-reconstructed target data of an 
% nlsaModel object.
%
% Modified 2020/08/29

x = getData( getKoopmanRecComponent( obj ), varargin{ : } );

