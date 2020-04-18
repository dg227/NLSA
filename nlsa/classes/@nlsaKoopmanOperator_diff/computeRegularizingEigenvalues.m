function eta = computeRegularizingEigenvalues( obj, diffOp )
% COMPUTEREGULARIZINGEIGENVALUES Compute eigenvalues of the regularizing 
% operator for an nlsaKoopmanOperator_diff object.
%
% Modified 2020/04/11

% Validate input arguments
if ~isa( diffOp, 'nlsaDiffusionOperator' ) || ~isscalar( diffOp )
    msgStr = [ 'Second input argument must be a scalar ' ...
               'nlsaDiffusionOperator object.' ];
    error( msgStr )
end

lambda = getEigenvalues( diffOp );       % kernel eigenvalues
idxPhi = getBasisFunctionIndices( obj ); % kernel eigenfunctions employed
if any( idxPhi > numel( lambda ) )
    error( 'Number of eigenvalues requested exceeds available eigenvalues' )
end

switch getRegularizationType( obj )

case 'lin'
    eta = ( 1 - lambda( idxPhi ) / lambda( 1 ) ) * lambda( 2 );

case 'log'
    if any( lambda( idxPhi ) < 0 ) 
        error( 'Negative eigenvalues detected in log normalization' )
    end
    eta = log( lambda( idxPhi ) );
    eta = eta / eta( 2 );

case 'inv'
    eta = lambda( 1 ) ./ lambda( idxPhi ) - 1;
    eta = eta / eta( 2 );

end


