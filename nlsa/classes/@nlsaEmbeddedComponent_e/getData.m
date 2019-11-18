function x = getData( obj, iB, varargin )
% GETDATA  Read data from nlsaEmbeddedComponent_o objects.
%
% This function can be called using either of the following formats:
%
% 1) x = getData( obj, iB, iR, iC, iA ), where obj is a scalar, vector, or 
%    matrix of nlsaEmbeddedComponent_o objects, returns the data stored in 
%    in batches iB, realizations iR, components iC, and "pages" iA in explicit
%    embedding format, using the same calling convention as the getData method
%    of the nlsaComponent class. 
%
% 2) x = getData( obj, iB, outFormat ), where obj is a scalar 
%    nlsaEmbeddedComponent_o object, and iB a positive scalar integer, 
%    returns the data from batch iB in the output format specified in the
%    string outFormat. outFormat can take the velues 'evector' or 'native'. 
%    'overlap' is not currently supported. 
%
% Modified 2019/11/17

if nargin == 3 && ischar( varargin{ 1 } )
    % Call method with specified output format
    x = getData_fmt( obj, iB, varargin{ 1 } );
else
    % Call method with standard calling syntax
    x = getData@nlsaComponent( obj, varargin{ : } );
end 




% GETDATA  Read data from an nlsaEmbeddedComponent_e object
%
% Modified 2019/11/17

if nargin == 2
    outFormat = 'evector'; 
end

if strcmp( outFormat, 'overlap' )
    error( 'Overlap output format not available' )
elseif ~any( strcmp( outFormat, { 'evector' 'native' } ) )
    error( 'Unrecognized output format' )
end

fileX = fullfile( getDataPath( obj ), getDataFile( obj, iB ) );
load( fileX, 'x' )
