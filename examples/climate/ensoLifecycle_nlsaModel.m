function [ model, In, Out ] = ensoLifecycle_nlsaModel( experiment )
% ENSOLIFECYCLE_NLSAMODEL Construct NLSA model for analysis of ENSO lifecycle
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
% to function climateNLSAModel_base to build the model.
%
% The constructed NLSA models have the following target components (used for
% composite lifecycle analysis):
%
% Component 1:  Nino 3.4 index
% Component 2:  Nino 4 index
% Component 3:  Nino 3 index
% Component 4:  Nino 1+2 index
% Component 5:  Global SST anomalies 
% Component 6:  Global SSH anomalies
% Component 7:  Global SAT anomalies
% Component 8:  Global precipitation rates
% Component 9:  Global surface zonal winds
% Component 10: Global surface meridional winds
% 
% Modified 2020/05/13

if nargin == 0
    experiment = 'noaa_industrial_IPSST_4yrEmb';
end

switch experiment

% NOAA 20th Century Reanalysis, industrial era, Indo-Pacific SST input,
% 4-year delay embeding window  
case 'noaa_industrial_IPSST_4yrEmb'
   
    % Dataset specification  
    In.Res( 1 ).dataset = 'noaa';                

    % Time specification
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '187001' '201906' }; % time limit  
    In.Res( 1 ).tCLim = { '198101' '201012' }; % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 1;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
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
    In.nPhi       = 501;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % Koopman generator parameters; in-sample data
    In.koopmanOpType = 'diff';     % Koopman generator type
    In.koopmanFDType  = 'central'; % finite-difference type
    In.koopmanFDOrder = 4;         % finite-difference order
    In.koopmanDt      = 1;         % sampling interval (in months)
    In.koopmanAntisym = true;      % enforce antisymmetrization
    In.koopmanEpsilon = 1E-3;      % regularization parameter
    In.koopmanRegType = 'inv';     % regularization type
    In.idxPhiKoopman  = 1 : 401;   % diffusion eigenfunctions used as basis
    In.nPhiKoopman    = numel( In.idxPhiKoopman ); % Koopman eigenfunctions to compute

% NOAA 20th Century Reanalysis, approximate satellite era, Indo-Pacific SST 
% input, 4-year delay embeding window  
case 'noaa_satellite_IPSST_4yrEmb'
    
    % Dataset specification 
    In.Res( 1 ).dataset = 'noaa';
    
    % Time specification 
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '197001' '201906' }; % time limit  
    In.Res( 1 ).tCLim = { '198101' '201012' }; % climatology time limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Batches to partition the in-sample data
    In.Res( 1 ).nB    = 1;          % partition batches
    In.Res( 1 ).nBRec = 1;          % batches for reconstructed data

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

    % Koopman generator parameters; in-sample data
    In.koopmanOpType = 'diff';     % Koopman generator type
    In.koopmanFDType  = 'central'; % finite-difference type
    In.koopmanFDOrder = 4;         % finite-difference order
    In.koopmanDt      = 1;         % sampling interval (in months)
    In.koopmanAntisym = true;      % enforce antisymmetrization
    In.koopmanEpsilon = 1E-3;      % regularization parameter
    In.koopmanRegType = 'inv';     % regularization type
    In.idxPhiKoopman  = 1 : 401;   % diffusion eigenfunctions used as basis
    In.nPhiKoopman    = numel( In.idxPhiKoopman );        % Koopman eigenfunctions to compute

% CCSM4 pre-industrial control, 200-year period, Indo-Pacific SST input, 4-year
% delay embeding window  
case 'ccsm4Ctrl_200yr_IPSST_4yrEmb'
   
    % Dataset specification  
    In.Res( 1 ).dataset = 'ccsm4Ctrl'; 

    % Time specification
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '019912' }; % time limit  
    In.Res( 1 ).tClim = In.Res( 1 ).tLim;     % climatology limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 1;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
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
    In.nPhi       = 501;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % Koopman generator parameters; in-sample data
    In.koopmanOpType = 'diff';     % Koopman generator type
    In.koopmanFDType  = 'central'; % finite-difference type
    In.koopmanFDOrder = 4;         % finite-difference order
    In.koopmanDt      = 1;         % sampling interval (in months)
    In.koopmanAntisym = true;      % enforce antisymmetrization
    In.koopmanEpsilon = 1E-3;      % regularization parameter
    In.koopmanRegType = 'inv';     % regularization type
    In.idxPhiKoopman  = 1 : 401;   % diffusion eigenfunctions used as basis
    In.nPhiKoopman    = numel( In.idxPhiKoopman ); % Koopman eigenfunctions to compute

% CCSM4 pre-industrial control, 1300-year period, Indo-Pacific SST input, 4-year
% delay embeding window  
case 'ccsm4Ctrl_1300yr_IPSST_4yrEmb'
   
    % Dataset specification  
    In.Res( 1 ).dataset = 'ccsm4Ctrl'; 

    % Time specification
    In.tFormat        = 'yyyymm';              % time format
    In.Res( 1 ).tLim  = { '000101' '130012' }; % time limit  
    In.Res( 1 ).tClim = In.Res( 1 ).tLim;     % climatology limits 

    % Source data specification 
    In.Src( 1 ).field = 'sstw';      % physical field
    In.Src( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Src( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; in-sample data
    In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
    In.Src( 1 ).nXB       = 1;          % samples before main interval
    In.Src( 1 ).nXA       = 0;          % samples after main interval
    In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
    In.Src( 1 ).fdType    = 'backward'; % finite-difference type
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
    In.nPhi       = 501;        % diffusion eigenfunctions to compute
    In.nPhiPrj    = In.nPhi;    % eigenfunctions to project the data
    In.idxPhiRec  = 1 : 1;      % eigenfunctions for reconstruction
    In.idxPhiSVD  = 1 : 1;      % eigenfunctions for linear mapping
    In.idxVTRec   = 1 : 1;      % SVD termporal patterns for reconstruction

    % Koopman generator parameters; in-sample data
    In.koopmanOpType = 'diff';     % Koopman generator type
    In.koopmanFDType  = 'central'; % finite-difference type
    In.koopmanFDOrder = 4;         % finite-difference order
    In.koopmanDt      = 1;         % sampling interval (in months)
    In.koopmanAntisym = true;      % enforce antisymmetrization
    In.koopmanEpsilon = 1E-3;      % regularization parameter
    In.koopmanRegType = 'inv';     % regularization type
    In.idxPhiKoopman  = 1 : 401;   % diffusion eigenfunctions used as basis
    In.nPhiKoopman    = numel( In.idxPhiKoopman ); % Koopman eigenfunctions to compute


otherwise
        error( 'Invalid experiment' )
end

%% PREPARE TARGET COMPONENTS (COMMON TO ALL MODELS)
%
% tStr is a string identifier for the analysis time interval.
%
% climStr is a string identifier for the climatology period relative to which
% anomalies are computed. 
%
% nETrg is the delay-embedding window for the target data

climStr = [ '_' In.Res( 1 ).tClim{ 1 } '-' In.Res( 1 ).tClim{ 2 } ];
tStr    = [ '_' In.Res( 1 ).tLim{ 1 } '-' In.Res( 1 ).tLim{ 2 } ];
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

% Global SST anomalies
In.Trg( 5 ).field = [ 'sstma' climStr ]; % physical field
In.Trg( 5 ).xLim  = [ 0 359 ];           % longitude limits
In.Trg( 5 ).yLim  = [ -89 89 ];          % latitude limits

% Global SSH anomalies
In.Trg( 6 ).field = [ 'sshma' climStr ]; % physical field
In.Trg( 6 ).xLim  = [ 0 359 ];          % longitude limits
In.Trg( 6 ).yLim  = [ -89 89 ] ;        % latitude limits

% Global SAT anomalies
In.Trg( 7 ).field = [ 'airma' climStr ]; % physical field
In.Trg( 7 ).xLim  = [ 0 359 ];           % longitude limits
In.Trg( 7 ).yLim  = [ -89 89 ];          % latitude limits
        
% Global precipitation anomalies
In.Trg( 8 ).field = [ 'pratema' climStr ]; % physical field
In.Trg( 8 ).xLim  = [ 0 359 ];             % longitude limits
In.Trg( 8 ).yLim  = [ -89 89 ];            % latitude limits

% Global surface zonal wind anomalies 
In.Trg( 9 ).field = [ 'uwndma' climStr ]; % physical field
In.Trg( 9 ).xLim  = [ 0 359 ];            % longitude limits
In.Trg( 9 ).yLim  = [ -89 89 ];           % latitude limits

% Global surface meridional wind anomalies
In.Trg( 10 ).field = [ 'vwndma' climStr ]; % physical field
In.Trg( 10 ).xLim  = [ 0 359 ];            % longitude limits
In.Trg( 10 ).yLim  = [ -89 89 ];           % latitude limits
    
% Abbreviated target component names
In.targetComponentName   = [ 'nino_sst_air_prate_uv' ];
In.targetRealizationName = '187001-201906';

% Prepare dalay-embedding parameters for target data
for iCT = 1 : numel( In.Trg )
        
    In.Trg( iCT ).idxE      = 1 : nETrg;  % delay embedding indices 
    In.Trg( iCT ).nXB       = 1;          % before main interval
    In.Trg( iCT ).nXA       = 0;          % samples after main interval
    In.Trg( iCT ).fdOrder   = 0;          % finite-difference order 
    In.Trg( iCT ).fdType    = 'backward'; % finite-difference type
    In.Trg( iCT ).embFormat = 'overlap';  % storage format
end


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
