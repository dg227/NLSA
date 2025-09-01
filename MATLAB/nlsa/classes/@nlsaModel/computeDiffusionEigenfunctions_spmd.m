function computeDiffusionEigenfunctions_spmd( obj, varargin )
% COMPUTEDIFFUSIONEIGENFUNCTIONS_SPMD Compute diffusion eigenfunctions of 
% nlsaModel objects with SPMD parallelization  
% 
% Modified 2014/06/30

Opt.ifComputeOperator = false;
Opt.ifWriteOperator   = false;
Opt.idxType           = 'double';
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataPhi.log';

diffOp = getDiffusionOperator( obj );
if Opt.ifComputeOperator
    sDist = getSymmetricDistance( obj );
    computeEigenfunctions_spmd( diffOp, sDist, ...
                               'logPath', getEigenfunctionPath( diffOp ), ...
                               'logFile', logFile, ...
                               'ifWriteOperator', Opt.ifWriteOperator, ...
                               'idxType', Opt.idxType );
else                           
     computeEigenfunctions_spmd( diffOp, ...
                               'logPath', getEigenfunctionPath( diffOp ), ...
                               'logFile', logFile, ...
                               'idxType', Opt.idxType );
end
