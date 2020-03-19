function x = getData( obj, varargin )
% GETDATA  Read data from nlsaEmbeddedComponent_o objects.
%
% This function can be called using either of the following formats:
%
% 1) x = getData( obj, iB, iR, iC, iA ), where obj is a scalar, vector, or 
%    matrix of nlsaEmbeddedComponent_o objects, returns the data stored in 
%    in batches iB, realizations iR, components iC, and pages iA in explicit
%    embedding format, using the same calling convention as the getData method
%    of the nlsaComponent class. 
%
% 2) x = getData( obj, iB, outFormat ), where obj is a scalar 
%    nlsaEmbeddedComponent_o object, and iB a positive scalar integer, 
%    returns the data from batch iB in the output format specified in the
%    string outFormat. outFormat can take the velues 'overlap', 'native', and
%    'evector', where in the former two cases the data is returned in 
%    'overlap' format, while in the latter case it is returned in explicit
%    embedding format. 
%
% Modified 2020/02/17

if nargin == 3 && ischar( varargin{ 2 } )
    % Call method with specified output format
    x = getData_fmt( obj, varargin{ : } );
else
    % Call method with standard calling syntax
    x = getData_std( obj, varargin{ : } );
end 

    

