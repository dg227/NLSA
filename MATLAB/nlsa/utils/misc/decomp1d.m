function [ idx nB ] = decomp1d( nS, nBatch, iBatch )
% DECOMP1D  Decompose a 1D domain of nS samples into nBatches for near-uniform
%           load. Modelled after MPI routine MPE_DECOMP1D.
%
% Modified 2012/12/05

switch nargin

    case 2
        idx = zeros( nBatch, 2 );
        nB  = zeros( nBatch, 1 );
        for iBatch = 1 : nBatch
            [ idx( iBatch, : ) nB( iBatch ) ] = decomp1d( nS, nBatch, iBatch );
        end

    case 3
        idx      = zeros( 1, 2 );
        nB       = floor( nS / nBatch );
        idx( 1 ) = ( iBatch - 1 ) * nB + 1;
        deficit  = mod( nS, nBatch );
        idx( 1 ) = idx( 1 ) + min( iBatch - 1, deficit );
        if iBatch - 1 < deficit
            nB = nB + 1;
        end
        idx( 2 ) = idx( 1 ) + nB - 1;
        if idx( 2 ) > nS || iBatch == nBatch 
            idx( 2 ) = nS;
        end
end    
