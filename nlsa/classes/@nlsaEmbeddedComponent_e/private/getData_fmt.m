function x = getData_fmt( obj, iB, outFormat )
%% GETDATA_FMT Read data from an nlsaEmbeddedComponent_e object in evector
%% (explicit delay embedding) format
%
% Modified 2019/11/17

if ~isscalar( obj )
    error( 'First input argument must be a scalar nlsaEmbeddedComponent_e object' )
end

if ~ispsi( iB ) || iB > getNBatch( obj )
    error( 'Second input argument must be a positive integer less than or equal to the number of samples in the partition of the first argument.' )
end

if nargin == 2
    outFormat = 'evector';
end

if ~any( strcmp( outFormat, { 'evector' 'native' } ) )
    error( 'Unrecognized output format' )
end

fileX = fullfile( getDataPath( obj ), getDataFile( obj, iB ) );
load( fileX, 'x' )

