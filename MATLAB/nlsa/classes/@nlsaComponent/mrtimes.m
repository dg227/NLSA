function c = mrtimes( obj, a );
% MRTIMES Right-multiply data in an array of nlsaComponent objects
% 
% Modified 2013/10/21


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validate object and numerical data arrays
if ~isnumeric( a )
    error( 'Second argument must be a numeric array' )
end
nS = sum( getNSample( obj( 1, : ) ) );
nD = sum( getDimension( obj( :, 1 )  ));
nA = size( a );
if nA( 1 ) ~= nS
    error( 'Incompatible numerical data array' )
end
nB         = getNBatch( obj( 1, :  ));
[ nC, nR ] = size( src );

c = zeros( nD, nA( 2 ) );
iD1 = 1;
for iC = 1 : nC 
    iD2 = iD1 + getDimension( obj( iC, 1 ) ) - 1;
    iS1 = 1;
    for iR = 1 : nR
        nSB = getBatchSize( obj( iC, iR ) );
        for iB = 1 : nB( iR )
            iS2 = iS1 + nSB - 1;
            c( iD1 : iD2, iS1 : iS2 ) = getData( obj( iC, iR ), iB ) * a( iD1 : iD2, iS1 : iS2 );
            iS1 = iS2 + 1;
        end
    end
    iD1 = iD2 + 1;
end
