function tag = concatenateRealizationTags( obj )
% CONCATENATEREALIZATIONTAGS Concatenate realization tags of a matrix of 
% nlsaComponent objects
%
% Modified 2019/11/15

if ~ismatrix( obj )
    error( 'Input argument must be a matrix of nlsaComponent objects.' )
end
nC = size( obj, 1 );

% If more than one rows (components) are present, operate recursively to 
% return a cell array of strings
if nC > 1 
    tag = cell( nC, 1 );
    for iC = 1 : nC
        tag{ iC } = concatenateRealizationTags( obj( iC, : ) );
    end
    return
end

% Default case of row-vector input argument
nR = size( obj, 2 );
tag  = cell( 1, nR );
for iR = 1 : nR
    tag{ iR } = getRealizationTag( obj( 1, iR ) );
end
tag = strjoin_e( tag, '_' );
