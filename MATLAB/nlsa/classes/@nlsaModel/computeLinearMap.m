function computeLinearMap( obj, varargin ) 
% COMPUTELINEARMAP Compute linear map(s) of nlsaModel objects 
% 
% Modified 2015/10/19

Opt.ifWriteOperator = true;
Opt.idxA            = 1 : numel( obj.linMap );
Opt                 = parseargs( Opt, varargin{ : } );


for iA = 1 : numel( Opt.idxA )
    computeLinearMap( obj.linMap( Opt.idxA( iA ) ), ...
                      obj.prjComponent, ... 
                      'logPath', ...
                      fullfile( getPath( obj.linMap( Opt.idxA ) ) ), ...
                      'logFile', 'dataA.log', ...
                      'ifWriteOperator', Opt.ifWriteOperator );
end
