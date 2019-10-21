function computeCovarianceOperator( obj, varargin )
% COMPUTECOVARIANCEOPERATOR Compute covariance operator of nlsaModel_ssa objects
% 
% Modified 2016/06/03

covOp = getCovarianceOperator( obj );
src   = getEmbComponent( obj );

logFile = 'dataP.log';

computeOperator( covOp, src, ...
                 'logPath', getOperatorPath( covOp ), ...
                 'logFile', logFile, ...
                 varargin{ : } );
