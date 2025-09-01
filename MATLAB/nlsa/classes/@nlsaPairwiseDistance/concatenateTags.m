function tag = concatenateTags( obj )
% CONCATENATETAGS Concatenate tags of a matrix of nlsaPairwiseDistance objects
%
% Modified 2015/10/27

tag = getTag( obj );
if numel( obj ) == 1
    tag = { tag };
end
tag = strjoin_e( tag( : ), '_' );
