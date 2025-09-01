function computeCovarianceEigenfunctions( obj, varargin )
% COMPUTECOVARIANCENEIGENFUNCTIONS Compute temporal covariancen eigenfunctions
%  of nlsaModel_ssa objects  
% 
% Modified 2016/06/02

Opt.ifComputeOperator = true;
Opt.ifWriteOperator   = false;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataV.log';

covOp = getCovarianceOperator( obj );
if Opt.ifComputeOperator
    src = getEmbComponent( obj );
    computeRightEigenfunctions( covOp, src, ...
                                'logPath', getRightSingularVectorPath( covOp ), ...
                                'logFile', logFile, ...
                                'ifWriteOperator', Opt.ifWriteOperator );
else                           
    computeRightEigenfunctions( covOp, ...
                               'logPath', getRightSingularVectorPath( covOp ), ...
                               'logFile', logFile );
end
