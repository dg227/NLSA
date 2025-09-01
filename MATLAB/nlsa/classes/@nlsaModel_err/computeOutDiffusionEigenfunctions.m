function computeOutDiffusionEigenfunctions( obj, varargin )
% COMPUTEMODDIFFUSIONEIGENFUNCTIONS Compute diffusion eigenfunctions 
% for the model data of an nlsaModel_err object  
% 
% Modified 2014/05/25

Opt.ifComputeOperator = false;
Opt.ifWriteOperator   = false;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataPhi.log';

diffOp = getOutDiffusionOperator( obj );
if Opt.ifComputeOperator
    sDist = getOutSymmetricDistance( obj );
    computeEigenfunctions( diffOp, sDist, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile, ...
                           'ifWriteOperator', Opt.ifWriteOperator );
else                           
     computeEigenfunctions( diffOp, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile );
end
