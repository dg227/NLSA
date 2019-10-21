function x = getOseReconstructedData( obj, varargin )
% GETOSERECONSTRUCTEDDATA Get OSE reconstructed target data of an 
% nlsaModel_den_ose object
%
% Modified 2018/07/01

x = getData( getOseRecComponent( obj ), varargin{ : } );

