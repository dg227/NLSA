function x = getOutData( obj, varargin )
% GETOUTDATA Get out-of-sample data of an nlsaModel_ose object
%
% Modified 2016/02/02

x = getData( getOutComponent( obj ), varargin{ : } );

