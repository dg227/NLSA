function a = getProjectedData( obj, iC, idxPhi )
% GETPROJECTEDDATA  Read projected data from a vector of 
% nlsaProjectedComponent objects
%
% Modified 2016/04/05

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
    idxPhi = 1 : getNBasisFunction( obj( 1 ) );
end

if any( idxPhi > getNBasisFunction( obj( 1 ) ) )
    error( 'Out of bounds basis function index' )
end

if isscalar( iC )
    file = fullfile( getProjectionPath( obj( iC ) ), ...
                     getProjectionFile( obj( iC ) ) ); 
    load( file, 'a' )
    a = a( :, idxPhi );
else
    nDE = getEmbeddingSpaceDimension( obj( iC ) );
    nPhi = numel( idxPhi );
    a = zeros( sum( nDE ), nPhi );
    iDE1 = 1;
    for i = 1 : numel( iC )
        iDE2 = iDE1 + nDE( i ) - 1;
        file = fullfile( getProjectionPath( obj( iC( i ) ) ), ...
                         getProjectionFile( obj( iC( i ) ) ) ); 
        A = load( file, 'a' );
        a( iDE1 : iDE2, : ) = A.a( :, idxPhi );
        iDE1 = iDE2 + 1;
    end
end

