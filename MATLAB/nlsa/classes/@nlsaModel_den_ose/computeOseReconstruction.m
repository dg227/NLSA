function computeOseReconstruction( obj, iCRec, iRRec, varargin )
% COMPUTEOSERECONSTRUCTION Reconstruct the OSE data from the projections onto
% the diffusion eigenfunctions of an nlsaModel_den_ose object
%
% Modified 2018/07/04


oseEmbComponent = getOseEmbComponent( obj );
recComponent    = getOseRecComponent( obj );
nCT             = size( oseEmbComponent, 1 );
nRO             = size( oseEmbComponent, 2 );

if nargin == 1
    iCRec = 1;
    iRRec = 1;
end

Opt.batch = 1 : getNBatch( getPartition( recComponent( iCRec, iRRec ) ) );
Opt       = parseargs( Opt, varargin{ : } );
logFile   = 'dataX_rec.log';

computeData( recComponent( iCRec, iRRec ), oseEmbComponent( iCRec, iRRec ), iCRec, iRRec, ...
             'batch', Opt.batch, ...
             'logPath', getDataPath( recComponent( iCRec, iRRec  ) ), ...
             'logFile', logFile );


