function eta = computeRegularizingEigenvalues( obj, diffOp )
% COMPUTEREGULARIZINGEIGENVALUES Compute eigenvalues of the regularizing 
% operator for an nlsaKoopmanOperator_diff object.
%
% Modified 2020/07/22

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

if any( lambda( idxPhi ) > 1 )
    error( 'Regularizing operator must have all eigenvalues <= 1.' )
end 
if any( lambda( idxPhi ) < 0 ) 
    error( 'Regularizing operator must have positive eigenvalues.' )
end

switch getRegularizationType( obj )

case 'lin'
    eta = 1 - lambda( idxPhi );

case 'log'
    eta = log( lambda( idxPhi ) );

case 'inv'
    eta = 1 ./ lambda( idxPhi ) - 1;
    eta = eta / eta( 2 );

end

iNrm = find( eta > 0, 1, 'first' );
eta = eta / eta( iNrm );
