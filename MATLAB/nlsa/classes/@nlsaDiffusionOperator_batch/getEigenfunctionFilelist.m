function file = getEigenfunctionFilelist( obj, iR )
% GETEIGENFUNCTIONFILELIST  Get eigenfunction filelist of an 
% nlsaDiffusionOperator object
%
% Modified 2014/04/19

if nargin == 1
    iR = 1 : numel( obj.filePhi );
end

for i = numel( iR ) : -1 : 1
    file( i ) = obj.filePhi( iR( i ) );
end

