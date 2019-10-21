function ifC = isCompatible( obj, cmp )
% ISCOMPATIBLE Check compatibility of nlsaFilelist objects 
%
% Modified 2014/06/13

switch class( cmp )
    case 'nlsaPartition'
        ifC  = true;
        sizF = size( obj );
        sizP = size( cmp );
        if numel( sizF ) ~= numel( sizP ) ...
          || any( sizF ~= sizP )
            ifC = false;
            return
        end
        ifC = all( getNFile( obj ) == getNBatch( cmp ) );
end
