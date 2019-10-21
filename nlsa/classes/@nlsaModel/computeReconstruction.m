function computeReconstruction( obj, iCRec, iRRec, varargin )
% COMPUTERECONSTRUCTION Reconstruct the target data from the projections onto
% the diffusion eigenfunctions of an nlsaModel object
%
% Modified 2016/04/04


trgEmbComponent = getTrgEmbComponent( obj );
prjComponent    = getPrjComponent( obj );
diffOp          = getDiffusionOperator( obj);
recComponent    = getRecComponent( obj );
nCT             = size( trgEmbComponent, 1 );
nR              = size( trgEmbComponent, 2 );

if nargin == 1
    iCRec = 1;
end

if nargin <= 2
    iRRec = 1;
end

Opt.batch = 1 : getNBatch( getPartition( recComponent( iCRec, iRRec ) ) );
Opt       = parseargs( Opt, varargin{ : } );
logFile   = 'dataX_rec.log';

computeData( recComponent( iCRec, iRRec ), prjComponent, diffOp, iCRec, iRRec, ...
             'batch', Opt.batch, ...
             'logPath', getDataPath( recComponent( iCRec, iRRec  ) ), ...
             'logFile', logFile );


