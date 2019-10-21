function setLeftSingularVectors( obj, uIn, iC, varargin )
% SETLEFTSINGULARVECTORS  Set left singular vector  data of an 
% nlsaCovarianceOperator object
%
% Modified 2015/10/19

nD = getDimension( obj );
nDU = sum( nD( iC ) );
nC = numel( nD );
if nargin == 2
    iC = 1 : nC;
end
nV = getNEigenfunction( obj );
if size( uIn, 1 ) ~= nDU
    error( 'Incompatible dimension samples' )
end
if size( uIn, 2 ) ~= min( nV, nDU ) 
    error( 'Incompatible number of singular vectors' )
end


iU1 = 1;
for i = 1 : numel( iC );
    iU2 = iU1 + nD( iC( i ) ) - 1;
    u = uIn( iU1 : iU2, : );
    file = fullfile( getLeftSingularVectorPath( obj ), ... 
                     getLeftSingularVectorFile( obj, iC ) );
    save( file, 'u', varargin{ : } )
end


