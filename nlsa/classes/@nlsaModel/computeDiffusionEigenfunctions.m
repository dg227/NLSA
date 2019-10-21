function computeDiffusionEigenfunctions( obj, varargin )
% COMPUTEDIFFUSIONEIGENFUNCTIONS Compute diffusion eigenfunctions of nlsaModel objects  
% 
% Modified 2015/05/08

Opt.ifComputeOperator = true;
Opt.ifWriteOperator   = false;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataPhi.log';

diffOp = getDiffusionOperator( obj );
if Opt.ifComputeOperator
    sDist = getSymmetricDistance( obj );
    computeEigenfunctions( diffOp, sDist, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile, ...
                           'ifWriteOperator', Opt.ifWriteOperator );
else                           
     computeEigenfunctions( diffOp, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile );
end
