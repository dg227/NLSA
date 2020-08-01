function [ model, In, Out ] = ensoQMDA_nlsaModel( experiment )
% ENSOQMDA_NLSAMODEL Construct NLSA model for quantum mechanical data 
% assimilation of ENSO.
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
% The constructed NLSA models have the following target components (used for
% prediction):
%
% Component 1:  Nino 3.4 index
% Component 2:  Nino 4 index
% Component 3:  Nino 3 index
% Component 4:  Nino 1+2 index
% 
% Modified 2020/08/01

if nargin == 0
    experiment = 'ersstV4_50yr_10yr_IPSST_2yrEmb_coneKernel';
end

switch experiment


% ERSSTv4 data, global domain, 1950-2010 for training, 2010-2020 
% for verification, 2-year delay embeding window, cone kernel  
case 'ersstV4_50yr_globalSST_4yrEmb_coneKernel'
    
    % Dataset specification 
    In.Res( 1 ).experiment = 'ersstV4';
    
    % Time specification (in-sample data) 
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '196001' '200912' }; % time limit  
    In.Res( 1 ).tClim = { '198101' '201012' }; % climatology time limits 

    % Time specification (out-of-sample data) 
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '200801' '202002' }; % time limit  
    Out.Res( 1 ).tClim = { '198101' '201012' }; % climatology time limits 

    trendStr = ''; % string identifier for detrening of target data

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 0 359 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -89 89 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 24;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 2;          % samples before main interval
    In.Src( 1 ).nXA       = 2;          % samples after main interval
    In.Src( 1 ).fdOrder   = 4;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'cone';     % local distance
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
    In.nPhi       = 451;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction


% ERSSTv4 data, Indo-Pacific domain, 1950-2010 for training, 2010-2020 
% for verification, 2-year delay embeding window, cone kernel  
case 'ersstV4_50yr_IPSST_4yrEmb_coneKernel'
    
    % Dataset specification 
    In.Res( 1 ).experiment = 'ersstV4';
    
    % Time specification (in-sample data) 
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '196001' '200912' }; % time limit  
    In.Res( 1 ).tClim = { '198101' '201012' }; % climatology time limits 

    % Time specification (out-of-sample data) 
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '200801' '202002' }; % time limit  
    Out.Res( 1 ).tClim = { '198101' '201012' }; % climatology time limits 

    trendStr = ''; % string identifier for detrening of target data

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 24;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 2;          % samples before main interval
    In.Src( 1 ).nXA       = 2;          % samples after main interval
    In.Src( 1 ).fdOrder   = 4;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central';  % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'cone';     % local distance
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
    In.nPhi       = 451;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction



% ERSSTv4 data, sub-global domain, 1950-2010 for training, 2010-2020 
% for verification, 2-year delay embeding window, cone kernel  
case 'ersstV4_satellite_subglobalSST_4yrEmb_coneKernel'
    
    % Dataset specification 
    In.Res( 1 ).experiment = 'ersstV4';
    
    % Time specification 
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '197801' '202002' }; % time limit  
    In.Res( 1 ).tClim = { '198101' '201012' }; % climatology time limits 

    trendStr = ''; % string identifier for detrening of target data

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 0 359 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -67 67 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 2;          % samples before main interval
    In.Src( 1 ).nXA       = 2;          % samples after main interval
    In.Src( 1 ).fdOrder   = 4;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central';  % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'cone';     % local distance
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
    In.nPhi       = 451;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction


% CCSM4 pre-industrial control, Indo-Pacific domain, 1100 years training, 200
% years test, 2-year delay-embeding window  
case 'ccsm4Ctrl_1100yr_200yr_IPSST_2yrEmb_coneKernel'
   
    % Dataset specification  
    In.Res( 1 ).experiment = 'ccsm4Ctrl'; 

    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = In.Res( 1 ).tLim;     % climatology limits 

    % Time specification (in-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = { '000101' '109912' }; % climatology limits  

    trendStr = ''; % string identifier for detrening of target data

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 24;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 2;          % samples before main interval
    In.Src( 1 ).nXA       = 2;          % samples after main interval
    In.Src( 1 ).fdOrder   = 4;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central';  % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'cone';     % local distance
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
    In.nPhi       = 501;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    
   
% CCSM4 pre-industrial control, global domain, 1100 years training, 200
% years test, 2-year delay-embeding window  
case 'ccsm4Ctrl_1100yr_200yr_globalSST_2yrEmb_coneKernel'

    % Dataset specification  
    In.Res( 1 ).experiment = 'ccsm4Ctrl'; 

    % Time specification (in-sample data)
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '109912' }; % time limit  
    In.Res( 1 ).tClim = In.Res( 1 ).tLim;     % climatology limits 

    % Time specification (in-sample data)
    Out.tFormat        = 'yyyymm';              % time format
    Out.Res( 1 ).tLim  = { '110001' '130012' }; % time limit  
    Out.Res( 1 ).tClim = { '000101' '109912' }; % climatology limits  

    trendStr = ''; % string identifier for detrening of target data

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 0 359 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -89 89 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 24;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 2;          % samples before main interval
    In.Src( 1 ).nXA       = 2;          % samples after main interval
    In.Src( 1 ).fdOrder   = 4;          % finite-difference order 
    In.Src( 1 ).fdType    = 'central'; % finite-difference type
    In.Src( 1 ).embFormat = 'overlap';  % storage format 

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1; % partition batches
    In.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % Batches to partition the out-of-sample data
    Out.Res( 1 ).nB    = 1; % partition batches
    Out.Res( 1 ).nBRec = 1; % batches for reconstructed data

    % NLSA parameters; in-sample data 
    In.nN         = 0;          % nearest neighbors; defaults to max. value if 0
    In.lDist      = 'cone';     % local distance
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
    In.nPhi       = 501;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction


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
    In.Trg( iCT ).nXB       = 1;          % before main interval
    In.Trg( iCT ).nXA       = 0;          % samples after main interval
    In.Trg( iCT ).fdOrder   = 0;          % finite-difference order 
    In.Trg( iCT ).fdType    = 'backward'; % finite-difference type
    In.Trg( iCT ).embFormat = 'overlap';  % storage format
end

% Prepare out-of-sample source and target data
Out.Src = In.Src;
Out.Trg = In.Trg;

%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );

%% SERIAL DATE NUMBERS FOR IN-SAMPLE DATA
% Loop over the in-sample realizations
for iR = 1 : numel( In.Res )
    limNum = datenum( In.Res( iR ).tLim, In.tFormat );
    nS = months( limNum( 1 ), limNum( 2 ) ) + 1; 
    In.Res( iR ).tNum = datemnth( limNum( 1 ), 0 : nS - 1 ); 
end

%% SERIAL DATE NUMBERS FOR OUT-OF-SAMPLE DATA
if ifOse
    % Loop over the out-of-sample realizations
    for iR = 1 : numel( Out.Res )
        limNum = datenum( Out.Res( iR ).tLim, Out.tFormat );
        nS = months( limNum( 1 ), limNum( 2 ) ) + 1; 
        Out.Res( iR ).tNum = datemnth( limNum( 1 ), 0 : nS - 1 ); 
    end
end

%% CONSTRUCT NLSA MODEL
if ifOse
    args = { In Out };
else
    args = { In };
end
[ model, In, Out ] = climateNLSAModel( args{ : } );
