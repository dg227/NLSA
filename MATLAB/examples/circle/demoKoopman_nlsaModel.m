function [ model, In, Out ] = demoKoopman_nlsaModel( experiment )
% DEMOKOOPMAN_NLSAMODEL Construct NLSA model for Koopman spectral  analysis 
% of variable-speed flow on the circle.
%
% Input arguments:
%
% experiment: A string identifier for the data analysis experiment. 
%
% Output arguments:
%
% model: Constructed model, in the form of an nlsaModel object.  
% In:    Data structure with in-sample model parameters. 
% Out:   Data structure with out-of-sample model parameters (optional). 
%
% This function creates the data structures In and Out, which are then passed 
% to function circleNLSAModel to build the model.
%
% Modified 2020/07/11
if nargin == 0
    experiment = 'a0.7';
end

switch experiment

    case 'a0.7'
        % In-sample dataset parameters
        In.Res.dt       = 0.01;      % sampling interval
        In.Res.f        = 1;         % frequency parameter
        In.Res.a        = 0.7;       % nonlinearity parameter
        In.Res.nSProd   = 6400;      % number of "production" samples
        In.Res.nSSpin   = 0;         % spinup samples
        In.Res.r1       = 1;         % ellipse axis 1    
        In.Res.r2       = 1;         % ellipse axis 2    
        In.Res.ifCenter = false;     % data centering

        % Source data
        In.Src.idxX    = 1 : 2;       % observed state vector components 
        In.Src.idxE    = 1 : 1;       % delay embedding indices
        In.Src.nXB     = 0;           % additional samples before main interval
        In.Src.nXA     = 0;         % additional samples after main interval
        In.Src.fdOrder = 0;           % finite-difference order 
        In.Src.fdType    = 'central'; % finite-difference type
        In.Src.embFormat = 'overlap'; % storage format for delay embedding

        % Target data
        In.Trg.idxX      = 1 : 2;     % observed state vector components 
        In.Trg.idxE      = 1 : 1;     % delay-embedding indices
        In.Trg.nXB       = 0;         % additional samples before main interval
        In.Trg.nXA       = 0;       % additional samples after main interval
        In.Trg.fdOrder   = 0;         % finite-difference order 
        In.Trg.fdType    = 'central'; % finite-difference type
        In.Trg.embFormat = 'evector'; % storage format for delay embedding

        % NLSA parameters
        In.Res.nB     = 1;          % batches to partition the in-sample data
        In.Res.nBRec  = 1;          % batches for reconstructed data
        In.nN         = 1500;       % nearest neighbors for pairwise distances
        In.lDist      = 'l2';       % local distance
        In.tol        = 0;          % 0 distance threshold (for cone kernel)
        In.zeta       = 0;          % cone kernel parameter 
        In.coneAlpha  = 0;          % velocity exponent in cone kernel
        In.nNS        = In.nN;      % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon     = 1;         % kernel bandwidth parameter 
        In.epsilonB    = 2;         % kernel bandwidth base
        In.epsilonE    = [ -20 20 ];% kernel bandwidth exponents 
        In.nEpsilon    = 200;       % number of exponents for bandwidth tuning
        In.alpha       = .5;        % diffusion maps normalization 
        In.nPhi        = 101;       % diffusion eigenfunctions to compute
        In.nPhiPrj     = In.nPhi;   % eigenfunctions to project the data
        In.idxPhiRec   = 1 : 1;     % eigenfunctions for reconstruction
        In.idxPhiSVD   = 1 : 1;     % eigenfunctions for linear mapping
        In.idxVTRec    = 1 : 5;     % SVD termporal patterns for reconstruction

        % NLSA parameters, kernel density estimation (KDE)
        In.denType      = 'vb';          % density estimation type
        In.denND        = 1;             % manifold dimension 
        In.denLDist     = 'l2';          % local distance function 
        In.denBeta      = -1 / In.denND; % density exponent 
        In.denNN        = 80;            % nearest neighbors 
        In.denZeta      = 0;             % cone kernel parameter 
        In.denConeAlpha = 0;             % cone kernel velocity exponent 
        In.denEpsilon   = 1;             % kernel bandwidth
        In.denEpsilonB  = 2;             % kernel bandwidth base 
        In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
        In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

        % Koopman generator parameters; in-sample data
        In.koopmanOpType = 'diff';     % Koopman generator type
        In.koopmanFDType  = 'central'; % finite-difference type
        In.koopmanFDOrder = 4;         % finite-difference order
        In.koopmanDt      = In.Res.dt; % sampling interval 
        In.koopmanAntisym = true;      % enforce antisymmetrization
        In.koopmanEpsilon = 1E-3;      % regularization parameter
        In.koopmanRegType = 'inv';     % regularization type
        In.idxPhiKoopman  = 1 : 11;   % diffusion eigenfunctions used as basis
        In.nPhiKoopman    = numel( In.idxPhiKoopman ); % Koopman eigenfunctions to compute
        In.nKoopmanPrj    = In.nPhiKoopman; % Koopman eigenfunctions for projection

    otherwise
        error( 'Invalid experiment' )
end


%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );

%% CONSTRUCT NLSA MODEL
if ifOse
    args = { In Out };
else
    args = { In };
end
[ model, In, Out ] = circleNLSAModel( args{ : } );
