function b = getProjectedVelocity( obj, iC, idxPhi )
% GETPROJECTEDVELOCITY  Read projected velocity data from a vector of 
% nlsaProjectedComponent_xi objects
%
% Modified 2014/06/26

if ~isCompatible( obj )
    error( 'Incompatible projected components' )
end
nC = numel( obj );

if nargin < 2
    iC = 1 : nC;
end

if any( iC > nC )
    error( 'Out of bounds component index' )
end

if nargin < 3
    idxPhi = 0 : getNBasisFunction( obj );
end

if any( idxPhi > getNBasisFunction( obj ) )
    error( 'Out of bounds basis function index' )
end

idxPhi = idxPhi + 1;
if isscalar( iC )
    file = fullfile( getVelocityProjectionPath( obj ), ...
                     getVelocityProjectionFile( obj ) ); 
    load( file, 'b' )
    b = b( :, idxPhi );
else
    nDE  = getEmbeddingSpaceDimension( obj( iC ) );
    nPhi = numel( idxPhi );
    b = zeros( sum( nDE ), nPhi );
    iDE1 = 1;
    for i = 1 : numel( iC )
        iDE2 = iDE1 + nDE( i ) - 1;
        file = fullfile( getVelocityProjectionPath( obj( iC( i ) ) ), ...
                         getVelocityProjectionFile( obj( iC( i ) ) ) ); 
        B = load( file, 'b' );
        b( iDE1 : iDE2, : ) = B.b( :, idxPhi );
        iDE1 = iDE2 + 1;
    end
end

