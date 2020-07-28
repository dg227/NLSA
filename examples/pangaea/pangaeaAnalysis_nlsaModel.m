function [ model, In, Out ] = pangaeaAnalysis_nlsaModel( experiment )
% PANGAEAANALYSIS_NLSAMODEL Construct NLSA model for analysis of PANGAEA data.
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
% This function creates the parameter structures In and Out, which are then 
% passed to function climateNLSAModel to build the model.
%
% Modified 2020/07/27

In.tFormat = 'yyyymm'; % time format

if nargin == 0
    experiment = 'temp_100kaEmb_l2Kernel';
end

switch experiment

case 'temp_100kaEmb_l2Kernel'

    % Time specification
    In.Res( 1 ).tLim  = [ -3000 0 ]; % time limit  

    % Source data
    In.Src( 1 ).field      = 'temp';     % physical field

    % Delay-embedding/finite-difference parameters
    In.Src( 1 ).idxE      = 1 : 150;    % delay-embedding indices 
    In.Src( 1 ).nXB       = 2;          % samples before main interval
    In.Src( 1 ).nXA       = 2;          % samples after main interval
    In.Src( 1 ).fdOrder   = 4;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central';  % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'l2';       % local distance
    In.tol        = 0;          % 0 distance threshold (for cone kernel)
    In.zeta       = 0.995;      % cone kernel parameter 
    In.coneAlpha  = 0;          % velocity exponent in cone kernel
    In.nNS        = In.nN;      % nearest neighbors for symmetric distance
    In.diffOpType = 'gl_mb_bs'; % diffusion operator type
    In.epsilon    = 2;          % kernel bandwidth parameter 
    In.epsilonB   = 2;          % kernel bandwidth base
    In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    In.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    In.alpha      = 0.5;        % diffusion maps normalization 
    In.nPhi       = 101;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    In.denND        = 5;             % manifold dimension
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    In.denNN        = 80;            % nearest neighbors 
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
In.Trg = In.Src;

%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );

%% SERIAL DATE NUMBERS FOR IN-SAMPLE DATA
% Loop over the in-sample realizations
for iR = 1 : numel( In.Res )
    nS = In.Res( iR ).tLim( 2 ) - In.Res( iR ).tLim( 1 ) + 1;
    In.Res( iR ).tNum = In.Res( iR ).tLim( 1 ) : In.Res( iR ).tLim( 2 ); 
end

%% SERIAL DATE NUMBERS FOR OUT-OF-SAMPLE DATA
if ifOse
    % Loop over the out-of-sample realizations
    for iR = 1 : numel( Out.Res )
        nS = Out.Res( iR ).tLim( 2 ) - Out.Res( iR ).tLim( 1 ) + 1;
        Out.Res( iR ).tNum = Out.Res( iR ).tLim( 1 ) : Out.Res( iR ).tLim( 2 ); 
    end
end

%% CONSTRUCT NLSA MODEL
if ifOse
    args = { In Out };
else
    args = { In };
end
[ model, In, Out ] = pangaeaNLSAModel( args{ : } );
