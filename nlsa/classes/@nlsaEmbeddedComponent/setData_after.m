function setData( obj, x, iB, varargin )
% SETDATA  Set batch data after the main time interval for  an 
% nlsaEmbeddedComponent object 
%
% Modified 2014/04/04

setData( obj, x, getNBatch( obj ) + 1, varargin{ : } )
