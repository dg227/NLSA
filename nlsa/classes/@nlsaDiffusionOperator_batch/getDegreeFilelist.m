function file = getDegreeFilelist( obj, iR )
% GETDEGREEFILELIST  Get degree filelist of an nlsaDiffusionOperator_batch 
% object
%
% Modified 2014/04/17

if nargin == 1
    iR = 1 : numel( obj.fileD );
end

for i = numel( iR ) : -1 : 1
    file( i ) = obj.fileD( iR( i ) );
end

