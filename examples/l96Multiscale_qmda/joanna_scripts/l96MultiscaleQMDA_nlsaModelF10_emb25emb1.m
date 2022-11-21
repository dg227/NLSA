function [ model,    In, ...   
           modelObs, InObs, OutObs ] = l96MultiscaleQMDA_nlsaModel( experiment )
% L96MULTISCALEQMDA_NLSAMODEL Construct NLSA models for quantum mechanical data 
% assimilation of the 2-level (multiscale) L96 model.
% 
% Input arguments:
%
% experiment: A string identifier for the data analysis experiment, as generated
%             by the driver script l96MultiscaleQMDA. 
%
% Output arguments:
%
% model:    Constructed nlsaModel object for the source data.  
% In:       Data structure with in-sample model parameters for the source data. 
% modelObs: Constructed nlsaModel object for the observed data.
% InObs:    Data structure with in-sample model parameters for the observed 
%           data. 
% OutObs:   Data structure with out-of-sample model parameters for the 
%           observed data. 
%
% This function creates the parameter structures In, InObs, and OutObs, which 
% are then passed to function l96MultiscaleNLSAModel to build the model.
%
% Modified 2021/07/05

if nargin == 0
    experiment = 'F10.0_eps0.00781_dt0.05_nS1000_nSOut100_idxXSrc1-+1-9_idxXObs1-+1-9_emb1_l2_den'
end

switch experiment

% Periodic regime
case 'F5.0_eps0.00781_dt0.05_nS10000_nSOut1000_idxXSrc1-+1-9_idxXObs1-+1-9_emb1_l2_den'
        
    % In-sample dataset parameters
    In.nX           = 9;       % Number of slow variables
    In.nY           = 8;       % Number of fast variables per slow variable
    In.dt           = 0.05;    % Sampling interval
    In.Res.F        = 5;       % Forcing parameter
    In.Res.epsilon  = 1 / 128; % Timesscale parameter for fast variables
    In.Res.hX       = -0.8;    % Coupling parameter  (slow to fast) 
    In.Res.hY       = 1;       % Coupling parameter (fast to slow)
    In.Res.nSProd   = 10000;   % Number of "production" samples
    In.Res.nSSpin   = 10000;   % Number of spinup samples
    In.Res.x0       = 1;       % Initial conditions for slow variables 
    In.Res.y0       = 1;       % Initial conditions for fast variables 
    In.Res.absTol   = 1E-7;    % Absolute tolerance for ODE solver
    In.Res.relTol   = 1E-5;    % Relative tolerance for ODE solver
    In.Res.ifCenter = false;   % Data centering

    % Out-of-sample dataset parameters
    Out.nX           = 9;       % Number of slow variables
    Out.nY           = 8;       % Number of fast variables per slow variable
    Out.dt           = 0.05;    % Sampling interval
    Out.Res.F        = 5;       % Forcing parameter
    Out.Res.epsilon  = 1 / 128; % Timesscale parameter for fast variables
    Out.Res.hX       = -0.8;    % Coupling parameter  (slow to fast) 
    Out.Res.hY       = 1;       % Coupling parameter (fast to slow)
    Out.Res.nSProd   = 1000;    % Number of "production" samples
    Out.Res.nSSpin   = 10000;   % Number of spinup samples
    Out.Res.x0       = 1.2;     % Initial conditions for slow variables 
    Out.Res.y0       = 1.2;     % Initial conditions for fast variables 
    Out.Res.absTol   = 1E-7;    % Absolute tolerance for ODE solver
    Out.Res.relTol   = 1E-5;    % Relative tolerance for ODE solver
    Out.Res.ifCenter = false;  % Data centering

    % Source data specification
    In.Src.idxX  = 1 : In.nX;   

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observations data specification
    In.Obs.idxX  = 1 : In.nX;   

    % Delay-embedding/finite-difference parameters; observations data
    In.Obs( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    In.Obs( 1 ).nXB       = 0;          % samples before main interval
    In.Obs( 1 ).nXA       = 0;          % samples after main interval
    In.Obs( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Obs( 1 ).fdType    = 'backward'; % finite-difference type
    In.Obs( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nParNN     = 0;          % parallel workers for nearest neighbors
    In.nParE      = 0;          % workers for delay-embedding sums
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'l2';       % local distance
    In.tol        = 0;          % 0 distance threshold (for cone kernel)
    In.zeta       = 0.995;      % cone kernel parameter 
    In.coneAlpha  = 1;          % velocity exponent in cone kernel
    In.nNS        = In.nN;      % nearest neighbors for symmetric distance
    In.diffOpType = 'gl_mb_bs'; % diffusion operator type
    In.epsilon    = 2;          % kernel bandwidth parameter 
    In.epsilonB   = 2;          % kernel bandwidth base
    In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    In.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    In.alpha      = 0.5;        % diffusion maps normalization 
    In.nPhi       = 1000;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    In.denND        = 5;             % manifold dimension
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    In.denNN        = 48;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

% Quasiperiodic regime
case 'F6.9_eps0.00781_dt0.05_nS10000_nSOut1000_idxXSrc1-+1-9_idxXObs1-+1-9_emb1_l2_den'
        
    % In-sample dataset parameters
    In.nX           = 9;       % Number of slow variables
    In.nY           = 8;       % Number of fast variables per slow variable
    In.dt           = 0.05;    % Sampling interval
    In.Res.F        = 6.9;     % Forcing parameter
    In.Res.epsilon  = 1 / 128; % Timesscale parameter for fast variables
    In.Res.hX       = -0.8;    % Coupling parameter  (slow to fast) 
    In.Res.hY       = 1;       % Coupling parameter (fast to slow)
    In.Res.nSProd   = 10000;   % Number of "production" samples
    In.Res.nSSpin   = 10000;   % Number of spinup samples
    In.Res.x0       = 1;       % Initial conditions for slow variables 
    In.Res.y0       = 1;       % Initial conditions for fast variables 
    In.Res.absTol   = 1E-7;    % Absolute tolerance for ODE solver
    In.Res.relTol   = 1E-5;    % Relative tolerance for ODE solver
    In.Res.ifCenter = false;   % Data centering

    % Out-of-sample dataset parameters
    Out.nX           = 9;       % Number of slow variables
    Out.nY           = 8;       % Number of fast variables per slow variable
    Out.dt           = 0.05;    % Sampling interval
    Out.Res.F        = 6.9;     % Forcing parameter
    Out.Res.epsilon  = 1 / 128; % Timesscale parameter for fast variables
    Out.Res.hX       = -0.8;    % Coupling parameter  (slow to fast) 
    Out.Res.hY       = 1;       % Coupling parameter (fast to slow)
    Out.Res.nSProd   = 1000;    % Number of "production" samples
    Out.Res.nSSpin   = 10000;   % Number of spinup samples
    Out.Res.x0       = 1.2;     % Initial conditions for slow variables 
    Out.Res.y0       = 1.2;     % Initial conditions for fast variables 
    Out.Res.absTol   = 1E-7;    % Absolute tolerance for ODE solver
    Out.Res.relTol   = 1E-5;    % Relative tolerance for ODE solver
    Out.Res.ifCenter = false;  % Data centering

    % Source data specification
    In.Src.idxX  = 1 : In.nX;   

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observations data specification
    In.Obs.idxX  = 1 : In.nX;   

    % Delay-embedding/finite-difference parameters; observations data
    In.Obs( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    In.Obs( 1 ).nXB       = 0;          % samples before main interval
    In.Obs( 1 ).nXA       = 0;          % samples after main interval
    In.Obs( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Obs( 1 ).fdType    = 'backward'; % finite-difference type
    In.Obs( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nParNN     = 0;          % parallel workers for nearest neighbors
    In.nParE      = 0;          % workers for delay-embedding sums
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'l2';       % local distance
    In.tol        = 0;          % 0 distance threshold (for cone kernel)
    In.zeta       = 0.995;      % cone kernel parameter 
    In.coneAlpha  = 1;          % velocity exponent in cone kernel
    In.nNS        = In.nN;      % nearest neighbors for symmetric distance
    In.diffOpType = 'gl_mb_bs'; % diffusion operator type
    In.epsilon    = 2;          % kernel bandwidth parameter 
    In.epsilonB   = 2;          % kernel bandwidth base
    In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    In.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    In.alpha      = 0.5;        % diffusion maps normalization 
    In.nPhi       = 1000;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    In.denND        = 5;             % manifold dimension
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    In.denNN        = 48;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

% Chaotic regime
case 'F10.0_eps0.00781_dt0.05_nS100000_nSOut30000_idxXSrc1-+1-9_idxXObs1-+1-9_emb25_l2_den'
        
    % In-sample dataset parameters
    In.nX           = 9;       % Number of slow variables
    In.nY           = 8;       % Number of fast variables per slow variable
    In.dt           = 0.05;    % Sampling interval
    In.Res.F        = 10;      % Forcing parameter
    In.Res.epsilon  = 1 / 128; % Timesscale parameter for fast variables
    In.Res.hX       = -0.8;    % Coupling parameter  (slow to fast) 
    In.Res.hY       = 1;       % Coupling parameter (fast to slow)
    In.Res.nSProd   = 100000;    % Number of "production" samples
    In.Res.nSSpin   = 10000;    % Number of spinup samples
    In.Res.x0       = 1;       % Initial conditions for slow variables 
    In.Res.y0       = 1;       % Initial conditions for fast variables 
    In.Res.absTol   = 1E-7;    % Absolute tolerance for ODE solver
    In.Res.relTol   = 1E-5;    % Relative tolerance for ODE solver
    In.Res.ifCenter = false;   % Data centering

    % Out-of-sample dataset parameters
    Out.nX           = 9;       % Number of slow variables
    Out.nY           = 8;       % Number of fast variables per slow variable
    Out.dt           = 0.05;    % Sampling interval
    Out.Res.F        = 10;      % Forcing parameter
    Out.Res.epsilon  = 1 / 128; % Timesscale parameter for fast variables
    Out.Res.hX       = -0.8;    % Coupling parameter  (slow to fast) 
    Out.Res.hY       = 1;       % Coupling parameter (fast to slow)
    Out.Res.nSProd   = 30000;     % Number of "production" samples
    Out.Res.nSSpin   = 10000;    % Number of spinup samples
    Out.Res.x0       = 1.2;     % Initial conditions for slow variables 
    Out.Res.y0       = 1.2;     % Initial conditions for fast variables 
    Out.Res.absTol   = 1E-7;    % Absolute tolerance for ODE solver
    Out.Res.relTol   = 1E-5;    % Relative tolerance for ODE solver
    Out.Res.ifCenter = false;  % Data centering

    % Source data specification
    In.Src.idxX  = 1 : In.nX;   

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 25;      % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observations data specification
    In.Obs.idxX  = 1 : In.nX;   

    % Delay-embedding/finite-difference parameters; observations data
    In.Obs( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    In.Obs( 1 ).nXB       = 0;          % samples before main interval
    In.Obs( 1 ).nXA       = 0;          % samples after main interval
    In.Obs( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Obs( 1 ).fdType    = 'backward'; % finite-difference type
    In.Obs( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nParNN     = 0;          % parallel workers for nearest neighbors
    In.nParE      = 0;          % workers for delay-embedding sums
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'l2';       % local distance
    In.tol        = 0;          % 0 distance threshold (for cone kernel)
    In.zeta       = 0.995;      % cone kernel parameter 
    In.coneAlpha  = 1;          % velocity exponent in cone kernel
    In.nNS        = In.nN;      % nearest neighbors for symmetric distance
    In.diffOpType = 'gl_mb_bs'; % diffusion operator type
    In.epsilon    = 2;          % kernel bandwidth parameter 
    In.epsilonB   = 2;          % kernel bandwidth base
    In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    In.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    In.alpha      = 0.5;        % diffusion maps normalization 
    In.nPhi       = 1000;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    In.denND        = 5;             % manifold dimension
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    In.denNN        = 48;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

otherwise
    error( 'Invalid experiment' )
end


%% PREPARE TARGET COMPONENTS (COMMON TO ALL MODELS)
for iCT = In.nX : -1 : 1 
    In.Trg( iCT ).idxX      = iCT;
    In.Trg( iCT ).idxE      = 1;         % delay-embedding indices
    In.Trg( iCT ).nXB       = 0;         % samples before main interval
    In.Trg( iCT ).nXA       = 0;         % samples after main interval
    In.Trg( iCT ).fdOrder   = 0;         % finite-difference order 
    In.Trg( iCT ).fdType    = 'central'; % finite-difference type
    In.Trg( iCT ).embFormat = 'evector'; % storage format for delay embedding
end

In.targetComponentName   = idx2str( 1 : In.nX, 'idxX' );
In.targetRealizationName = '_';

%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );
if ifOse
    Out.Trg = In.Trg;
end


%% CONSTRUCT NLSA MODELS FOR SOURCE AND OBSERVED DATA
InObs = In;
InObs.Src = InObs.Obs;
OutObs = Out;

[ model, In ] = l96MultiscaleNLSAModel( In );

if ifOse
    args = { InObs OutObs };
else
    args = { InObs };
end
[ modelObs, InObs, OutObs ] = l96MultiscaleNLSAModel( args{ : } );
