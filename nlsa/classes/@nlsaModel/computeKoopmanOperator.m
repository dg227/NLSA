function computeKoopmanOperator( obj, varargin )
% COMPUTEKOOPMANOPERATOR Compute Koopman operator of nlsaModel object
% 
% Modified 2020/04/15

koopmanOp = getKoopmanOperator( obj );
diffOp    = getDiffusionOperator( obj );

logFile = 'dataV.log';

computeOperator( koopmanOp, diffOp, ...
                 'logPath', getOperatorPath( koopmanOp ), ...
                 'logFile', logFile, ...
                 varargin{ : } );
