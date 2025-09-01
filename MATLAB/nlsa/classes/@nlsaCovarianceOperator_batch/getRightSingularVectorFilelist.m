function file = getRightSingularVectorFilelist( obj, iR )
% GETRIGHTSINGULARVECTORFILELIST  Get right singular vector filelist of an 
% nlsaCovariance operator object
%
% Modified 2014/07/16


if nargin == 1
    iR = 1 : numel( obj.fileV );
end

for i = numel( iR ) : -1 : 1
    file( i ) = obj.fileV( iR( i ) );
end

