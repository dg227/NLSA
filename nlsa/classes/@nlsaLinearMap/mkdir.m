function mkdir( obj ) 
% MKDIR Make directories of nlsaLinearMap objects
%
% Modified 2015/10/19

mkdir@nlsaCovarianceOperator_gl( obj )

for iObj = 1 : numel( obj )
    pth = getTemporalPatternPath( obj( iObj ) );
    if ~isdir( pth )
        mkdir( pth )
    end
end
