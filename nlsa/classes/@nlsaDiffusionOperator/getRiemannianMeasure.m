function mu = getRiemannianMeasure( obj )
% GETRIEMANNIANMEASURE  Read the Riemannian measure of an nlsaDiffusionOperator object
%
% Modified 2014/02/10

file = fullfile( getOperatorPath( obj ), getEigenfunctionFile( obj ) );
load( file, 'mu' )
