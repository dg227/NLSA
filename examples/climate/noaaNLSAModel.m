function [ model, In, Out ] = noaaNLSAModel( experiment )
% NOAANLSAMODEL Construct NLSA model for NOAA 20th Century reanalysis (20CRv2)
%  data.
% 
%  experiment is a string identifier of the data analysis experiment.
%
%  In and Out are data structures containing the in-sample (training) and 
%  out-of-sample (verification) model parameters, respectively.
%
%  This function creates a data structure In and (optionally) a data structure
%  Out, which are then passed to the function climateNLSAModel_base to 
%  construct the model.
%
%  The following function is provided for data import: 
%
%      noaaData
%               
%  Longitude range is [ 0 359 ] at 1 degree increments
%  Latitude range is [ -89 89 ] at 1 degree increments
%  Date range is is January 1854 to June 2019  at 1 month increments
%
% Modified 2020/04/21
 
if nargin == 0
    experiment = 'enso_lifecycle';
end

switch experiment

    % ENSO LIFECYCLE BASED ON INDO-PACIFIC SST 
    % Source (covariate) data is area-weighted Indo-Pacific SST
    %
    % Target (response) data is:
    %
    % Component 1: Nino 3.4 index
    % Component 2: Global SST anomalies 
    % Component 3: Global SAT anomalies
    % Component 4: Global precipitation rates
    case 'enso_lifecycle'

        % In-sample dataset parameters 
        In.tFormat             = 'yyyymm';              % time format
        In.Res( 1 ).tLim       = { '187001' '201906' }; % time limit  
        In.Res( 1 ).experiment = 'noaa';                % 20CRv2 dataset
        In.Src( 1 ).field      = 'sstw';      % physical field
        In.Src( 1 ).xLim       = [ 28 290 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -60  20 ]; % latitude limits
        In.Trg( 1 ).field      = 'sstmawav_198101-201012';  % physical field
        In.Trg( 1 ).xLim       = [ 190 240 ];  % longitude limits
        In.Trg( 1 ).yLim       = [ -5 5 ];     % latitude limits
        In.Trg( 2 ).field      = 'sstma_198101-201012';  % physical field
        In.Trg( 2 ).xLim       = [ 0 359 ];   % longitude limits
        In.Trg( 2 ).yLim       = [ -89 89 ];  % latitude limits
        In.Trg( 3 ).field      = 'airma_198101-201012';  % physical field
        In.Trg( 3 ).xLim       = [ 0 359 ];   % longitude limits
        In.Trg( 3 ).yLim       = [ -89 89 ];  % latitude limits
        In.Trg( 4 ).field      = 'pratema_198101-201012';  % physical field
        In.Trg( 4 ).xLim       = [ 0 359 ];   % longitude limits
        In.Trg( 4 ).yLim       = [ -89 89 ];  % latitude limits
        In.Trg( 5 ).field      = 'uwndma_198101-201012';  % physical field
        In.Trg( 5 ).xLim       = [ 0 359 ];   % longitude limits
        In.Trg( 5 ).yLim       = [ -89 89 ];  % latitude limits
        In.Trg( 6 ).field      = 'vwndma_198101-201012';  % physical field
        In.Trg( 6 ).xLim       = [ 0 359 ];   % longitude limits
        In.Trg( 6 ).yLim       = [ -89 89 ];  % latitude limits
        
        % Abbreviated target component names
        In.targetComponentName = [ 'sstw_sstmawav_air_prate_uv' ];
        In.targetRealizationName = '187001-201906';

        % Delay-embedding/finite-difference parameters; in-sample data
        In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
        In.Src( 1 ).nXB       = 1;          % samples before main interval
        In.Src( 1 ).nXA       = 0;          % samples after main interval
        In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap';  % storage format 
        In.Trg( 1 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;          % before main interval
        In.Trg( 1 ).nXA       = 0;          % samples after main interval
        In.Trg( 1 ).fdOrder   = 1;          % finite-difference order 
        In.Trg( 1 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap';  % storage format
        In.Trg( 2 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 2 ).nXB       = 1;          % before main interval
        In.Trg( 2 ).nXA       = 0;          % samples after main interval
        In.Trg( 2 ).fdOrder   = 1;          % finite-difference order 
        In.Trg( 2 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 2 ).embFormat = 'overlap';  % storage format
        In.Trg( 3 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 3 ).nXB       = 1;          % before main interval
        In.Trg( 3 ).nXA       = 0;          % samples after main interval
        In.Trg( 3 ).fdOrder   = 1;          % finite-difference order 
        In.Trg( 3 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 3 ).embFormat = 'overlap';  % storage format
        In.Trg( 4 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 4 ).nXB       = 1;          % before main interval
        In.Trg( 4 ).nXA       = 0;          % samples after main interval
        In.Trg( 4 ).fdOrder   = 1;          % finite-difference order 
        In.Trg( 4 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 4 ).embFormat = 'overlap';  % storage format
        In.Trg( 5 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 5 ).nXB       = 1;          % before main interval
        In.Trg( 5 ).nXA       = 0;          % samples after main interval
        In.Trg( 5 ).fdOrder   = 1;          % finite-difference order 
        In.Trg( 5 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 5 ).embFormat = 'overlap';  % storage format
        In.Trg( 6 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 6 ).nXB       = 1;          % before main interval
        In.Trg( 6 ).nXA       = 0;          % samples after main interval
        In.Trg( 6 ).fdOrder   = 1;          % finite-difference order 
        In.Trg( 6 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 6 ).embFormat = 'overlap';  % storage format
        In.Res( 1 ).nB        = 1;          % partition batches
        In.Res( 1 ).nBRec     = 1;          % batches for reconstructed data

        % NLSA parameters; in-sample data 
        In.nN         = 0; % nearest neighbors; defaults to max. value if 0
        In.lDist      = 'cone'; % local distance
        In.tol        = 0;      % 0 distance threshold (for cone kernel)
        In.zeta       = 0.995;  % cone kernel parameter 
        In.coneAlpha  = 0;      % velocity exponent in cone kernel
        In.nNS        = In.nN;  % nearest neighbors for symmetric distance
        In.diffOpType = 'gl_mb_bs'; % diffusion operator type
        In.epsilon    = 2;        % kernel bandwidth parameter 
        In.epsilonB   = 2;          % kernel bandwidth base
        In.epsilonE   = [ -40 40 ]; % kernel bandwidth exponents 
        In.nEpsilon   = 200;      % number of exponents for bandwidth tuning
        In.alpha      = 0.5;      % diffusion maps normalization 
        In.nPhi       = 501;      % diffusion eigenfunctions to compute
        In.nPhiPrj    = In.nPhi;  % eigenfunctions to project the data
        In.idxPhiRec  = 1 : 1;    % eigenfunctions for reconstruction
        In.idxPhiSVD  = 1 : 1;    % eigenfunctions for linear mapping
        In.idxVTRec   = 1 : 1;    % SVD termporal patterns for reconstruction

        % Koopman generator parameters; in-sample data
        In.koopmanOpType = 'diff';      % Koopman generator type
        In.koopmanFDType  = 'central'; % finite-difference type
        In.koopmanFDOrder = 4;         % finite-difference order
        In.koopmanDt      = 1;         % sampling interval (in months)
        In.koopmanAntisym = true;     % enforce antisymmetrization
        In.koopmanEpsilon = 1E-3;      % regularization parameter
        In.koopmanRegType = 'inv';     % regularization type
        In.idxPhiKoopman  = 1 : 401;    % diffusion eigenfunctions used as basis
        In.nPhiKoopman    = numel( In.idxPhiKoopman );        % Koopman eigenfunctions to compute

    otherwise
        error( 'Invalid experiment' )
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

[ model, In, Out ] = climateNLSAModel_base( args{ : } );
