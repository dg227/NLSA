function setData_before( obj, x, varargin )
% SETDATA  Set batch data before the main time interval for an 
% nlsaEmbeddedComponent object 
%
% Modified 2014/04/04

setData( obj, x, 0, varargin{ : } )
