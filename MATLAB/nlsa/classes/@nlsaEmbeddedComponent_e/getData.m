function x = getData( obj, varargin )
% GETDATA  Read data from nlsaEmbeddedComponent_e objects.
%
% This function can be called using either of the following formats:
%
% 1) x = getData( obj, iB, iR, iC, iA ), where obj is a scalar, vector, matrix, 
%    or 3D array of nlsaEmbeddedComponent_e objects, returns the data stored in 
%    in batches iB, realizations iR, components iC, and pages iA in explicit
%    embedding format, using the same calling convention as the getData method
%    of the nlsaComponent class. 
%
% 2) x = getData( obj, iB, outFormat ), where obj is a scalar 
%    nlsaEmbeddedComponent_e object, and iB a positive scalar integer, 
%    returns the data from batch iB in the output format specified in the
%    string outFormat. outFormat can take the velues 'evector' or 'native'. 
%    'overlap' is not currently supported. 
%
% Modified 2020/01/29

if nargin == 3 && ispsi( varargin{ 1 } ) && ischar( varargin{ 2 } )
    % Call method with specified output format
    x = getData_fmt( obj, varargin{ 1 }, varargin{ 2 } );
else
    % Call method with standard calling syntax
    x = getData@nlsaComponent( obj, varargin{ : } );
end 
