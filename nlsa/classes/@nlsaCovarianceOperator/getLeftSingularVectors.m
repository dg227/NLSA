function [ u, s ] = getLeftSingularVectors( obj, iC )
% GETLEFTSINGULARVECTORS  Read left singular vectors of an 
% nlsaCovarianceOperator object
%
% Modified 2015/10/19

nD = getDimension( obj );
nC = numel( nD );

if nargin < 2
    iC = 1 : nC;
end

varNames = { 'u' };
if isscalar( iC )
    file = fullfile( getLeftSingularVectorPath( obj ), ...
                     getLeftSingularVectorFile( obj, iC ) );
    load( file, varNames{ : } )
else
    nU = sum( nD( iC ) );
    u = zeros( nU, getNEigenfunction( obj ) );
    iU1 = 1;
    for i = 1 : numel( iC )
        iU2 = iU1 + nD( iC( i ) ) - 1;
        file = fullfile( getLeftSingularVectorPath( obj ), ...
                         getLeftSingularVectorFile( obj, iC( i ) ) );
        B = load( file, varNames{ : } );
        v( iU1 : iU2, : ) = B.u;
        iU1 = iU2 + 1;
    end
end

if nargout == 2
    s = getSingularValues( obj );
end
