function c = mltimes( obj, a )
% MLTIMES Left-multiply data in an array of nlsaComponent objects
% 
% Modified 2013/10/24

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate object and numerical data arrays
if ~isnumeric( a )
    error( 'Second argument must be a numeric array' )
end
nS = sum( getNSample( obj( 1, : ) ) );
nD = getEmbeddingSpaceDimension( obj( :, 1 ) );
nA = size( a ); 
if nA( 1 ) ~= sum( nD )
    error( 'Incompatible numerical data array' )
end

nB         = getNBatch( obj( 1, :  ));
[ nC, nR ] = size( obj );

c = zeros( nS, nA( 2 ) );

iS1 = 1;
for iR = 1 : nR
    nSB = getBatchSize( obj( 1, iR ) );
    for iB = 1 : nB( iR )
        iS2 = iS1 + nSB - 1;
        iD1 = 1;
        for iC = 1 : nC
            iD2 = iD1 + nD( iC ) - 1;
            c( iS1 : iS2, : )  = c( iS1 : iS2, : ) ...
                               +   getData( obj( iC, iR ), iB )' * a( iD1 : iD2, : );
            iD1 = iD2 + 1;
        end
        iS1 = iS2 + 1;
    end
end
