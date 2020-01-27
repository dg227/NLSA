function x = getData_fmt( obj, iB, outFormat )
%% GETDATA_FMT Read data from an nlsaEmbeddedComponent_e object in evector
%% (explicit delay embedding) format
%
% Modified 2020/01/25

if ~isscalar( obj )
    error( 'First input argument must be a scalar nlsaEmbeddedComponent_e object' )
end

if ~isnnsi( iB ) || iB > getNBatch( obj ) + 1
    error( 'Second input argument must be a non-negative scalar integer less than or equal to 1 plus the number of batches in the partition of the first argument.' )
end

if nargin == 2
    outFormat = 'evector';
end

if ~any( strcmp( outFormat, { 'evector' 'native' } ) )
    error( 'Unrecognized output format' )
end

fileX = fullfile( getDataPath( obj ), getDataFile( obj, iB ) );
load( fileX, 'x' )

