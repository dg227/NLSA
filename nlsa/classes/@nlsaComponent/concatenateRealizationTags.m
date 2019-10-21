function tag = concatenateRealizationTags( obj )
% CONCATENATEREALIZATIONTAGS Concatenate realization tags of a matrix of nlsaEmbeddedComponent objects
%
% Modified 2014/07/29

nR = size( obj, 2 );
tag  = cell( 1, nR );
for iR = 1 : nR
    tag{ iR } = getRealizationTag( obj( 1, iR ) );
end
