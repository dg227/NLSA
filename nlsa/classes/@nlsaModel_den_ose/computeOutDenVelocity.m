function computeOutDenVelocity( obj, iProcC, nProcC, iProcR, nProcR, varargin )
% COMPUTEOuTDENVELOCITY Compute phase space velocity for the out-of-sample  
% density data of an nlsaModel_den_ose object
%
% Modified 2018/07/04

if nargin == 1 || isempty( iProcC )
    iProcC = 1;
    nProcC = 1;
    iProcR = 1;
    nProcR = 1;
end

Opt.ifWriteXi = true;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataXi.log';

emb = getOseDenEmbComponent( obj );
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

