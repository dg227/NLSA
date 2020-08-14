%experiment = 'ccsm4Ctrl_1300yr_IPSSTA_0yrEmb_l2Kernel';
experiment = 'ccsm4Ctrl_1300yr_IPSST_4yrEmb_coneKernel';

nPhi = 501;
nQ = 48;

model = ensoLifecycle_nlsaModel( experiment );
[ phi, mu, lambda ] = getDiffusionEigenfunctions( model );

phiMu    = phi( 1 : end - nQ, 1 : nPhi ) .* mu( 1 : end - nQ );
phiShift = phi( nQ + 1 : end, 1 : nPhi );  

U = phiMu' * phiShift;
UK = U .* lambda( 1 : nPhi )'; 

R = ( UK + UK' ) / 2;

[ c, xi ] = eig( R );
xi = diag( xi );
[ xi, idx ] = sort( xi, 'descend' );
c = c( :, idx );

psi = phi( :, 1 : nPhi ) * c;
