function file = getOperatorFilelist( obj, iR )
% GETOPERATORFILELIST  Get operator filelist of an nlsaDiffusionOperator_batch 
% object
%
% Modified 2014/04/18


if nargin == 1
    iR = 1 : numel( obj.fileP );
end

for i = numel( iR ) : -1 : 1
    file( i ) = obj.fileP( iR( i ) );
end

