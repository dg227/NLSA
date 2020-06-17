function a = getKoopmanProjectedData( obj, varargin )
% GETKOOPMANPROJECTEDDATA Get projected target data onto the Koopman 
% eigenfunctions of an nlsaModel object
%
% Modified 2020/06/16

a = getProjectedData( getKoopmanPrjComponent( obj ), varargin{ : } );

