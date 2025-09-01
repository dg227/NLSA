function x = getData_after( obj, varargin )
% GETDATA_AFTER  Read data after main interval from nlsaEmbeddedComponent_e 
% objects.
%
% This function can be called using either of the following formats:
%
% 1) x = getData_after( obj, iR, iC, iA ), where obj is a scalar, vector,
%    matrix, or 3D array of nlsaEmbeddedComponent_e objects, returns the data
%    after the main time interval in realizations iR, components iC, and pages
%    iA in explicit embedding format. 
%
% 2) x = getData_after( obj, outFormat ), where obj is a scalar 
%    nlsaEmbeddedComponent_e object returns the data from after the main time
%    interval in the output format specified in the string outFormat. 
%    outFormat can take the velues 'overlap', 'native', and
%    'evector', where in the former two cases the data is returned in 
%    'overlap' format, while in the latter case it is returned in explicit
%    embedding format. 
%
% Modified 2020/03/19

if nargin == 2 && ischar( varargin{ 1 } )
    % Call method with specified output format
    x = getData_after_fmt( obj, varargin{ : } );
else
    % Call method with standard calling syntax
    x = getData_after_std( obj, varargin{ : } );
end 

    

