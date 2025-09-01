function obj = setNormalizationSubpath( obj, pth )
% SETNORMALIZATIONSUBPATH  Set normalization subdirectory of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2014/04/09

if ~isrowstr( pth )
    error( 'Path must be a character string' )
end
obj.pathQ = pth;
