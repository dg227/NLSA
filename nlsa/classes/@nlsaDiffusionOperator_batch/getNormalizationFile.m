function file = getNormalizationFile( obj, iB, iR )
% GETDNORMALIZATIONFILE  Get normalization filename from an 
% nlsaDiffusionOperator_batch object 
%
% Modified 2014/04/09

if nargin == 2 
     [ iB, iR ] = gl2loc( getPartition( obj ), iB );
end

file = getFile( getNormalizationFilelist( obj, iR ), iB );
