function a = getProjectedData( obj, varargin )
% GETDPROJECTEDDATA Get projected target data onto the covariance eigenfunction
% of an nlsaModel_ssa object
%
% Modified 2016/05/31

a = getProjectedData( getPrjComponent( obj ), varargin{ : } );

