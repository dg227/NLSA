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
    experiment = 'ccsm4Ctrl_000101-130012_IPSST_IPSST_emb12_l2_den';
end

switch experiment

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
    In.Obs( 1 ).field = 'sstw';      % physical field
    In.Obs( 1 ).xLim  = [ 28 290 ];  % longitude limits
    In.Obs( 1 ).yLim  = [ -60  20 ]; % latitude limits

    % Delay-embedding/finite-difference parameters; source data
    % We add samples in In.Obs( 1 ).nXB to align the observed data with the 
    % center of the embedding window used in the source data.
    In.Obs( 1 ).idxE      = 1 : 5;      % delay-embedding indices 
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
    In.nPhi       = 3001;       % diffusion eigenfunctions to compute
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


%% CHECK IF WE ARE DOING OUT-OF-SAMPLE EXTENSION
ifOse = exist( 'Out', 'var' );
if ifOse
    Out.Trg = In.Trg;
end


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

%% CONSTRUCT NLSA MODELS FOR SOURCE AND OBSERVED DATA
InObs = In;
InObs.Src = InObs.Obs;
OutObs = Out;

[ model, In ] = climateNLSAModel( In );

if ifOse
    args = { InObs OutObs };
else
    args = { InObs };
end
[ modelObs, InObs, OutObs ] = climateNLSAModel( args{ : } );
