function computeSVDReconstruction( obj, iCRec, iRRec, varargin )
% COMPUTESVDRECONSTRUCTION Reconstruct the target data from the projections onto
% the SVD temporal patterns of an nlsaModel_ssa object
%
% Modified 2016/05/31


trgEmbComponent = getTrgEmbComponent( obj );
linMap          = getLinearMap( obj);
svdRecComponent = getSvdRecComponent( obj );
nCT             = size( trgEmbComponent, 1 );
nR              = size( trgEmbComponent, 2 );

if nargin == 1
    iCRec = 1;
    iRRec = 1;
end

Opt.batch = 1 : getNBatch( getPartition( svdRecComponent( iCRec, iRRec ) ) );
Opt       = parseargs( Opt, varargin{ : } );
logFile   = 'dataX_rec.log';

computeData( svdRecComponent, linMap, linMap, iCRec, iRRec, ...
             'batch', Opt.batch, ...
             'logPath', getDataPath( svdRecComponent( iCRec, iRRec  ) ), ...
             'logFile', logFile );


