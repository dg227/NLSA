function computeSVDTemporalPatterns( obj, varargin )
% COMPUTESVDTEMPORALPATTERNS Compute SVD temporal patterns of an nlsaModel 
% object
%
% Modified 2015/10/20

Opt.ifWritePatterns = true;
Opt = parseargs( Opt, varargin{ : } );

computeTemporalPatterns( getLinearMap( obj ), ...
                         getDiffusionOperator( obj ), ...
                         'ifWritePatterns', Opt.ifWritePatterns );

