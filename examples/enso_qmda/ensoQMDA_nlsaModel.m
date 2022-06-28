function [ model,    In, ...   
           modelObs, InObs, OutObs ] = ensoQMDA_nlsaModel( experiment )
% ENSOQMDA_NLSAMODEL Construct NLSA models for quantum mechanical data 
% assimilation of ENSO.
% 
% Input arguments:
%
% experiment: A string identifier for the data analysis experiment. The 
% identifier has the form
%
% <SourceDataset>_<TrainingDateRange>_<TestDateRange> ...
% _<SourceVariable>_<ObsVariable>  ...
% _emb<EmbeddingWindowLength>_<kernelType>_<densityFlag> 
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
% are then passed to function climateNLSAModel to build the model.
%
% The model for the source data has the following target components (used for
% prediction):
%
% Component 1:  Nino 3.4 index
% Component 2:  Nino 4 index
% Component 3:  Nino 3 index
% Component 4:  Nino 1+2 index
% 
% Modified 2021/07/05

if nargin == 0
    experiment = 'ccsm4Ctrl_000101-130012_IPSST_IPSST_emb11_l2_den';
end

switch experiment

case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST_IPSST_emb9_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 9;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 48;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST_IPSST_emb11_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 11;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 48;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST_IPSST_emb12_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 12;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 48;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning


case 'ccsm4Ctrl_000101-109912_110001-130012_IPSST_IPSST_emb13_cone'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 13;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 1;          % samples before main interval
    In.Src( 1 ).nXA       = 1;          % samples after main interval
    In.Src( 1 ).fdOrder   = 2;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central';  % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.lDist      = 'cone';     % local distance
    In.tol        = 0;          % 0 distance threshold (for cone kernel)
    In.zeta       = 0.995;      % cone kernel parameter 
    In.coneAlpha  = 1;          % velocity exponent in cone kernel
    In.nNS        = In.nN;      % nearest neighbors for symmetric distance
    In.diffOpType = 'gl';       % diffusion operator type
    In.epsilon    = 1;          % kernel bandwidth parameter 
    In.epsilonB   = 2;          % kernel bandwidth base
    In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    In.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    In.alpha      = 1;        % diffusion maps normalization 
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 48;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

case 'ccsm4Ctrl_000101-109912_110001-130012_Nino3.4_IPSST_emb11_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    climStr = [ '_' In.Res( 1 ).tClim{ 1 } '-' In.Res( 1 ).tClim{ 2 } ];

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = [ 'sstmawav' climStr ]; % physical field
    In.Src( 1 ).xLim  = [ 190 240 ];            % longitude limits
    In.Src( 1 ).yLim  = [ -5 5 ];               % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 11;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 48;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

case 'ccsm4Ctrl_000101-109912_110001-130012_Nino3.4_IPSST_emb17_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    climStr = [ '_' In.Res( 1 ).tClim{ 1 } '-' In.Res( 1 ).tClim{ 2 } ];

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = [ 'sstmawav' climStr ]; % physical field
    In.Src( 1 ).xLim  = [ 190 240 ];            % longitude limits
    In.Src( 1 ).yLim  = [ -5 5 ];               % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 17;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 100;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

case 'ccsm4Ctrl_000101-109912_110001-130012_Nino3.4_IPSST_emb23_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    climStr = [ '_' In.Res( 1 ).tClim{ 1 } '-' In.Res( 1 ).tClim{ 2 } ];

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = [ 'sstmawav' climStr ]; % physical field
    In.Src( 1 ).xLim  = [ 190 240 ];            % longitude limits
    In.Src( 1 ).yLim  = [ -5 5 ];               % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 23;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 1;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 100;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

case 'ccsm4Ctrl_000101-109912_110001-130012_Nino3.4_IPSST_emb47_l2_den'
    % Dataset specification (in-sample data)
    In.Res( 1 ).experiment = 'ccsm4Ctrl';
    
    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = { '000101' '109912' }; % climatology time limits 

    climStr = [ '_' In.Res( 1 ).tClim{ 1 } '-' In.Res( 1 ).tClim{ 2 } ];

    % Dataset specification (out-of-sample data)
    Out.Res( 1 ).experiment = 'ccsm4Ctrl';

    % Time specification (out-of-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = In.Res( 1 ).tClim;     % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = [ 'sstmawav' climStr ]; % physical field
    In.Src( 1 ).xLim  = [ 190 240 ];            % longitude limits
    In.Src( 1 ).yLim  = [ -5 5 ];               % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    In.Src( 1 ).idxE      = 1 : 47;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 0;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Observed data specification 
    InObs.Src( 1 ).field = 'sstw';      % physical field
    InObs.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    InObs.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in InObs.Src( 1 ).nXB to align the observed data with the 
    % the source data.
    InObs.Src( 1 ).idxE      = 1 : 5;      % delay-embedding indices 
    InObs.Src( 1 ).nXB       = In.Src( 1 ).nXB;  % samples before main interval
    InObs.Src( 1 ).nXA       = In.Src( 1 ).nXA;  % samples after main interval
    InObs.Src( 1 ).fdOrder   = 0;          % finite-difference order 
    InObs.Src( 1 ).fdType    = 'backward'; % finite-difference type
    InObs.Src( 1 ).embFormat = 'overlap';  % storage format 

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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
    % In.nPhi       = 4001;       % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    In.denType      = 'vb';          % density estimation type
    % In.denND        = 5;             % manifold dimension
    In.denND        = 6;
    In.denLDist     = 'l2';          % local distance function 
    In.denBeta      = -1 / In.denND; % density exponent 
    % In.denNN        = 48;             % nearest neighbors 
    In.denNN        = 400;             % nearest neighbors 
    In.denZeta      = 0;             % cone kernel parameter 
    In.denConeAlpha = 0;             % cone kernel velocity exponent 
    In.denEpsilon   = 1;             % kernel bandwidth
    In.denEpsilonB  = 2;             % kernel bandwidth base 
    In.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    In.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

    % NLSA parameters; observed data 
    InObs.nParNN     = 0;          % parallel workers for nearest neighbors
    InObs.nParE      = 0;          % workers for delay-embedding sums
    InObs.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    InObs.lDist      = 'l2';       % local distance
    InObs.tol        = 0;          % 0 distance threshold (for cone kernel)
    InObs.zeta       = 0.995;      % cone kernel parameter 
    InObs.coneAlpha  = 1;          % velocity exponent in cone kernel
    InObs.nNS        = InObs.nN;      % nearest neighbors for symmetric distance
    InObs.diffOpType = 'gl_mb_bs'; % diffusion operator type
    InObs.epsilon    = 2;          % kernel bandwidth parameter 
    InObs.epsilonB   = 2;          % kernel bandwidth base
    InObs.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
    InObs.nEpsilon   = 200;        % number of exponents for bandwidth tuning
    InObs.alpha      = 0.5;        % diffusion maps normalization 
    InObs.nPhi       = 3001;       % diffusion eigenfunctions to compute
    InObs.nPhiPrj    = InObs.nPhi;    % eigenfunctions to project the data
    InObs.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    InObs.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    InObs.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % NLSA parameters, kernel density estimation (KDE)
    InObs.denType      = 'vb';          % density estimation type
    InObs.denND        = 5;             % manifold dimension
    InObs.denLDist     = 'l2';          % local distance function 
    InObs.denBeta      = -1 / InObs.denND; % density exponent 
    InObs.denNN        = 100;             % nearest neighbors 
    InObs.denZeta      = 0;             % cone kernel parameter 
    InObs.denConeAlpha = 0;             % cone kernel velocity exponent 
    InObs.denEpsilon   = 1;             % kernel bandwidth
    InObs.denEpsilonB  = 2;             % kernel bandwidth base 
    InObs.denEpsilonE  = [ -20 20 ];    % kernel bandwidth exponents 
    InObs.denNEpsilon  = 200;       % number of exponents for bandwidth tuning

otherwise
    error( 'Invalid experiment' )
end

%% PREPARE TARGET COMPONENTS (COMMON TO ALL MODELS)
%
% climStr is a string identifier for the climatology period relative to which
% anomalies are computed. 
%
% nETrg is the delay-embedding window for the target data

climStr = [ '_' In.Res( 1 ).tClim{ 1 } '-' In.Res( 1 ).tClim{ 2 } ];
nETrg   = 1; 

% Nino 3.4 index
In.Trg( 1 ).field = [ 'sstmawav' climStr ]; % physical field
In.Trg( 1 ).xLim  = [ 190 240 ];            % longitude limits
In.Trg( 1 ).yLim  = [ -5 5 ];               % latitude limits

% Nino 4 index
In.Trg( 2 ).field = [ 'sstmawav' climStr ]; % physical field
In.Trg( 2 ).xLim  = [ 160 210 ];            % longitude limits
In.Trg( 2 ).yLim  = [ -5 5 ];               % latitude limits

% Nino 3 index
In.Trg( 3 ).field = [ 'sstmawav' climStr ]; % physical field
In.Trg( 3 ).xLim  = [ 210 270 ];            % longitude limits
In.Trg( 3 ).yLim  = [ -5 5 ];               % latitude limits

% Nino 1+2 index
In.Trg( 4 ).field = [ 'sstmawav' climStr ]; % physical field
In.Trg( 4 ).xLim  = [ 270 280 ];            % longitude limits
In.Trg( 4 ).yLim  = [ -10 0 ];              % latitude limits

% Abbreviated target component names
In.targetComponentName   = 'nino';
In.targetRealizationName = '_';

% Prepare dalay-embedding parameters for target data
for iCT = 1 : numel( In.Trg )
    In.Trg( iCT ).idxE      = 1 : nETrg;  % delay embedding indices 
    In.Trg( iCT ).nXB       = 0;          % before main interval
    In.Trg( iCT ).nXA       = 0;          % samples after main interval
    In.Trg( iCT ).fdOrder   = 0;          % finite-difference order 
    In.Trg( iCT ).fdType    = 'backward'; % finite-difference type
    In.Trg( iCT ).embFormat = 'overlap';  % storage format
end


%% SERIAL DATE NUMBERS
% Loop over the in-sample realizations
for iR = 1 : numel( In.Res )
    limNum = datenum( In.Res( iR ).tLim, In.tFormat );
    nS = months( limNum( 1 ), limNum( 2 ) ) + 1; 
    In.Res( iR ).tNum = datemnth( limNum( 1 ), 0 : nS - 1 ); 
end

% Loop over the out-of-sample realizations
for iR = 1 : numel( Out.Res )
    limNum = datenum( Out.Res( iR ).tLim, Out.tFormat );
    nS = months( limNum( 1 ), limNum( 2 ) ) + 1; 
    Out.Res( iR ).tNum = datemnth( limNum( 1 ), 0 : nS - 1 ); 
end

%% CONSTRUCT NLSA MODELS FOR SOURCE AND OBSERVED DATA
InObs.Res = In.Res;
InObs.Trg = In.Trg;
InObs.tFormat = In.tFormat;
InObs.targetComponentName = In.targetComponentName;
InObs.targetRealizationName = In.targetRealizationName;

Out.Trg = In.Trg;
OutObs = Out;

[ model, In ] = climateNLSAModel( In );
[ modelObs, InObs, OutObs ] = climateNLSAModel( InObs, OutObs );
