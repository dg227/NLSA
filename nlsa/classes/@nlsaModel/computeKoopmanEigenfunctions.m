function computeKoopmanEigenfunctions( obj, varargin )
% COMPUTEKOOPMANEIGENFUNCTIONS Compute Koopman eigenfunctions of nlsaModel 
% objects  
% 
% Modified 2020/04/15

Opt.ifComputeOperator = true;
Opt.ifWriteOperator   = false;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataZeta.log';

koopmanOp = getKoopmanOperator( obj );
if Opt.ifComputeOperator
    diffOp = getDiffusionOperator( obj );
    computeEigenfunctions( koopmanOp, diffOp, ...
                           'logPath', getEigenfunctionPath( koopmanOp ), ...
                           'logFile', logFile, ...
                           'ifWriteOperator', Opt.ifWriteOperator );
else                           
     computeEigenfunctions( koopmanOperator, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile );
end
