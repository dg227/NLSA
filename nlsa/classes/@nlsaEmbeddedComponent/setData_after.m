function setData_after( obj, x, varargin )
% SETDATA_AFTER  Set batch data after the main time interval for  an 
% nlsaEmbeddedComponent object 
%
% Modified 2020/03/18

setData( obj, x, getNBatch( obj ) + 1, varargin{ : } )
