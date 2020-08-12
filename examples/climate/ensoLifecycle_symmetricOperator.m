nPhi = 201;
nQ = 48;

model = ensoLifecycle_nlsaModel( experiment );
[ phi, mu, lambda ] = getDiffusionEigenfunctions( model );

phiMu    = phi( 1 : end - nQ, 1 : nPhi ) .* mu( 1 : end - nQ );
phiShift = phi( nQ + 1 : end, 1 : nPhi );  

U = phiMu' * phiShift;
UK = U .* lambda( 1 : nPhi )'; 

R = ( UK + UK' ) / 2;

[ c, xi ] = eig( R );
[ xi, idx ] = sort( xi, 'descend' );
c = c( :, idx );

psi = phi * c;
