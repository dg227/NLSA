function computeSclOutDiffusionEigenfunctions( obj, varargin )
% COMPUTESCLOUTDIFFUSIONEIGENFUNCTIONS Compute scaled diffusion eigenfunctions
% for the OS data of an nlsaModel_scl object 
% 
% Modified 2014/07/28

Opt.ifComputeOperator = false;
Opt.ifWriteOperator   = false;
Opt = parseargs( Opt, varargin{ : } );
logFile = 'dataPhi.log';

diffOp = getSclOutDiffusionOperator( obj );
if Opt.ifComputeOperator
    sDist = getSclOutSymmetricDistance( obj );
    computeEigenfunctions( diffOp, sDist, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile, ...
                           'ifWriteOperator', Opt.ifWriteOperator );
else                           
     computeEigenfunctions( diffOp, ...
                           'logPath', getEigenfunctionPath( diffOp ), ...
                           'logFile', logFile );
end
