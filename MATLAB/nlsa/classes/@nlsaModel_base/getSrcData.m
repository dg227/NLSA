function x = getSrcData( obj, varargin )
% GETSRCDATA Get source data of an nlsaModel_base object
%
% Modified 2016/02/02

x = getData( getSrcComponent( obj ), varargin{ : } );

