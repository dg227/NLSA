function computeDenVelocity( obj, iProcC, nProcC, iProcR, nProcR, varargin )
% COMPUTEDENVELOCITY Compute phase space velocity for the density data of 
% an nlsaModel_den object
%
% Modified 2014/12/29

if nargin == 1 || isempty( iProcC )
    iProcC = 1;
    nProcC = 1;
    iProcR = 1;
    nProcR = 1;
end

Opt.ifWriteXi = true;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataXi.log';

emb = getDenEmbComponent( obj );
[ nC, nR ] = size( emb );

pPartitionC = nlsaPartition( 'nSample', nC, 'nBatch', nProcC );
pPartitionR = nlsaPartition( 'nSample', nR, 'nBatch', nProcR );
iCLim = getBatchLimit( pPartitionC, iProcC );
iRLim = getBatchLimit( pPartitionR, iProcR );

for iR = iRLim( 1 ) : iRLim( 2 )
    for iC = iCLim( 1 ) : iCLim( 2 )
        computeVelocity(  emb( iC, iR ), ...
                   'logPath', getVelocityPath( emb( iC, iR ) ), ...
                   'logFile', logFile, ...
                   'ifWriteXi', Opt.ifWriteXi );
    end
end

