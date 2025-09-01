function x = getOutData( obj, varargin )
% GETOUTDATA Get out-of-sample data of an nlsaModel__den_ose object
%
% Modified 2018/07/01

x = getData( getOutComponent( obj ), varargin{ : } );

