function xE = lembed( x, iSLim, argE1, argE2 )
%LEMBED Time-lagged embedding
%
%   Modified 12/12/2012

switch nargin
    case 3
        % argE1 contains the indices for embedding
        idxE = argE1;
    case 4
        % argE1 is the embedding window length
        % argE2 is the embedding window sampling interval
        idxE = 1 : argE2 : argE1;
end

% Quick return if idxE == 1
if idxE == 1    
    xE = x( :, iSLim( 1 ) : iSLim( 2 ) );
    return
end

idxE = idxE( end ) + 1 - idxE;
nSE = numel( idxE );
nD  = size( x, 1 );
nDE = nD * nSE;
nS  = iSLim( 2 ) - iSLim( 1 ) + 1;
xE  = zeros( nDE, nS );

for iS = 1 : nS
    iSE         = iSLim( 1 ) - idxE + iS;
    xE( :, iS ) = reshape( x( :, iSE ), [ nDE 1 ] );
end
