function computeSVDTemporalPatterns( obj, varargin )
% COMPUTESVDTEMPORALPATTERNS Compute SVD temporal patterns of an nlsaModel_ssa 
% object
%
% Modified 2016/05/31

Opt.ifWritePatterns = true;
Opt = parseargs( Opt, varargin{ : } );

computeTemporalPatterns( getLinearMap( obj ), ...
                         getCovarianceOperator( obj ), ...
                         'ifWritePatterns', Opt.ifWritePatterns );

