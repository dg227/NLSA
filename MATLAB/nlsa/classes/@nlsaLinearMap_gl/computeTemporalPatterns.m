function [ vT, mu ] = computeTemporalPatterns( obj, kOp, varargin )
% COMPUTETEMPORALPATTERNS Compute temporal patterns associated with 
% nlsaLinearMap objects
% 
% Modified 2015/10/20

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do some basic checking of input arguments
if ~isscalar( obj )
    error( 'First argument must be scalar' )
end
if ~isscalar( kOp ) || ~isa( kOp, 'nlsaKernelOperator' )
    error( 'Eigenfunction data must be specified as a scalar nlsaKernelOperator object' )
end
if ~isCompatible( obj, kOp )
    error( 'Incompatible kernel operator' )
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parse optional input arguments
Opt.ifWritePatterns = true;
Opt = parseargs( Opt, varargin{ : } );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read eigenfunctions and compute temporal patterns
tic
[ vT, mu ] = getEigenfunctions( kOp, [], [], getBasisFunctionIndices( obj ) );
v  = getRightSingularVectors( obj );
vT = vT * v;  

if Opt.ifWritePatterns
    setTemporalPatterns( obj, vT, mu, '-v7.3' )
end
