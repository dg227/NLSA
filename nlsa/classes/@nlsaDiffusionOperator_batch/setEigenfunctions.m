function setEigenfunctions( obj, v, mu, iB, varargin )
% SETEIGENFUNCTIONS  Set eigenfunction data of an nlsaDiffusionOperator_batch
% object
%
% Modified 2014/07/21

partition = getPartition( obj );
if ~ischar( varargin{ 1 } )
    iR = varargin{ 1 }; % varargin{ 1 } stores realization
    varargin = varargin( 2 : end );
else
    [ iB, iR ]  = gl2loc( partition, iB );
end

nEig = getNEigenfunction( obj );
if size( v, 1 ) ~= getBatchSize( partition( iR ), iB )
    error( 'Incompatible number of samples' )
end
if size( v, 2 ) ~= nEig 
    error( 'Incompatible number of eigenfunctions' )
end

if ~isempty( mu )
    if ~isvector( mu ) || numel( mu ) ~= size( v, 1 )
        error( 'Invalid Riemannian measure' )
    end
    nSave = 2;
else
    nSave = 1;
end

varNames = { 'v' 'mu' };

file = fullfile( getEigenfunctionPath( obj ), ... 
                 getEigenfunctionFile( obj, iB, iR ) );
save( file, varNames{ 1 : nSave }, varargin{ : } )


