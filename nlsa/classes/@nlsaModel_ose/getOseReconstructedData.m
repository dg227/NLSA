function x = getOseReconstructedData( obj, varargin )
% GETOSERECONSTRUCTEDDATA Get OSE reconstructed target data of an 
% nlsaModel_ose object
%
% Modified 2015/12/14

x = getData( getOseRecComponent( obj ), varargin{ : } );

