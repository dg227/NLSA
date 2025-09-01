function computeSVD( obj, varargin ) 
% COMPUTESVD Singular value decomposition for nlsaModel_ssa objects 
% 
% Modified 2016/05/31

Opt.mode                         = 'calcA';
Opt.ifWriteOperator              = false;
Opt.ifWriteLeftSingularVectors   = true;
Opt.ifWriteSingularValues        = true;
Opt.ifWriteRightSingularVectors  = true;
Opt.idxA                         = 1 : numel( obj.linMap );
Opt                              = parseargs( Opt, varargin{ : } );

idxPhi = getBasisFunctionIndices( obj.linMap( end ) );


logFile = 'dataSVD.log';

computeSVD( obj.linMap( Opt.idxA ), obj.prjComponent, ...
            'mode',                        Opt.mode, ...
            'logPath',                     fullfile( getPath( obj.linMap( end ) ) ), ...
            'logFile',                     logFile, ...
            'ifWriteOperator',             Opt.ifWriteOperator, ...
            'ifWriteLeftSingularVectors',  Opt.ifWriteLeftSingularVectors, ...
            'ifWriteSingularValues',       Opt.ifWriteSingularValues, ...
            'ifWriteRightSingularVectors', Opt.ifWriteRightSingularVectors );
