function file = getNormalizationFilelist( obj, iR )
% GETNORMALIZATIONFILELIST  Get normalizaton filelist of an 
% nlsaDiffusionOperator_batch object
%
% Modified 2014/04/17


if nargin == 1
    iR = 1 : numel( obj.fileQ );
end

for i = numel( iR ) : -1 : 1
    file( i ) = obj.fileQ( iR( i ) );
end

