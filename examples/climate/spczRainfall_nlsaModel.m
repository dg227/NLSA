function [ model, In, Out ] = spczRainfall_nlsaModel( dataset, period, ...
    experiment )
% SPCZRAINFALL_NLSAMODEL Construct NLSA model for analysis of SPCZ rainfall
%
% Input arguments:
%
% dataset:    A string identifier for the dataset analyzed.
% period:     A string identifier for the analysis period.
% experiment: A string identifier for the data analysis experiment. 
%
% Output arguments:
%
% model: Constructed model, in the form of an nlsaModel object.  
% In:    Data structure with in-sample model parameters. 
% Out:   Data structure with out-of-sample model parameters (optional). 
%
% This function creates the data structures In and Out, which are then passed 
% to function climateNLSAModel_base to construct the model.
%
% Possible options for 
%
% Modified 2020/05/09

In.Res( 1 ).experiment = dataset; % data analysis product

In.tFormat = 'yyyymm'; % time format


switch dataset

%% NOAA/CMAP REANALYSIS DATA
case 'noaa'
    switch period
    case 'satellite'
        In.Res( 1 ).tLim    = { '197901' '201912' }; % time limits
    otherwise
        error( 'Invalid period.' )
    end

    switch experiment

    % SPCZ analysis using Indo-Pacific precipitation data
    case 'IPPrecip' 
        % In-sample data specification 

        % Indo-Pacific precip (source)
        In.Src( 1 ).field      = 'prate';     % physical field
        In.Src( 1 ).xLim       = [ 28 290 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -60  20 ]; % latitude limits

        % Indo-Pacifid precip anomalies (target) 
        In.Trg( 1 ).field      = 'prate';  % physical field
        In.Trg( 1 ).xLim       = [ 28 290 ];   % longitude limits
        In.Trg( 1 ).yLim       = [ -60 20 ];  % latitude limits

        % Delay-embedding/finite-difference parameters
        % Indo-Pacific precip source data
        In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
        In.Src( 1 ).nXB       = 1;          % samples before main interval
        In.Src( 1 ).nXA       = 0;          % samples after main interval
        In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap';  % storage format 

        % Indo-Pacific precip target data
        In.Trg( 1 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;          % before main interval
        In.Trg( 1 ).nXA       = 0;          % samples after main interval
        In.Trg( 1 ).fdOrder   = 0;          % finite-difference order 
        In.Trg( 1 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap';  % storage format

        % Batches to partition the in-sample data
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
        In.nPhi       = 101;      % diffusion eigenfunctions to compute
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
        In.idxPhiKoopman  = 1 : 51;    % diffusion eigenfunctions used as basis
        In.nPhiKoopman    = numel( In.idxPhiKoopman );  % Koopman eigenfunctions to compute

    % SPCZ analysis using Indo-Pacific precipitation data
    case 'PacPrecip' 
        % In-sample data specification 

        % Indo-Pacific precip (source)
        In.Src( 1 ).field      = 'prate';     % physical field
        In.Src( 1 ).xLim       = [ 135 270 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -35  35 ]; % latitude limits

        % Indo-Pacifid precip anomalies (target) 
        In.Trg( 1 ).field      = 'prate';  % physical field
        In.Trg( 1 ).xLim       = [ 28 290 ];   % longitude limits
        In.Trg( 1 ).yLim       = [ -60 20 ];  % latitude limits

        % Delay-embedding/finite-difference parameters
        % Indo-Pacific precip source data
        In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
        In.Src( 1 ).nXB       = 1;          % samples before main interval
        In.Src( 1 ).nXA       = 0;          % samples after main interval
        In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap';  % storage format 

        % Indo-Pacific precip target data
        In.Trg( 1 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;          % before main interval
        In.Trg( 1 ).nXA       = 0;          % samples after main interval
        In.Trg( 1 ).fdOrder   = 0;          % finite-difference order 
        In.Trg( 1 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap';  % storage format

        % Batches to partition the in-sample data
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
        In.nPhi       = 101;      % diffusion eigenfunctions to compute
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
        In.idxPhiKoopman  = 1 : 51;    % diffusion eigenfunctions used as basis
        In.nPhiKoopman    = numel( In.idxPhiKoopman );  % Koopman eigenfunctions to compute


    otherwise
        error( 'Invalid experiment' )
    end



%% CCSM4 pre-industrial control
case 'ccsm4Ctrl'
    switch period
    case '200yr'
        In.Res( 1 ).tLim = { '000101' '019912' }; 
    case '1300yr'
        In.Res( 1 ).tLim = { '000101' '130012' }; 
    otherwise
        error( 'Invalid period' )
    end

    switch experiment

    % SPCZ analysis using Indo-Pacific precipitation data
    case 'IPPrecip' 
        % In-sample data specification 

        % Indo-Pacific precip (source)
        In.Src( 1 ).field      = 'prate';     % physical field
        In.Src( 1 ).xLim       = [ 28 290 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -60  20 ]; % latitude limits

        % Indo-Pacifid precip anomalies (target) 
        In.Trg( 1 ).field      = 'prate';  % physical field
        In.Trg( 1 ).xLim       = [ 28 290 ];   % longitude limits
        In.Trg( 1 ).yLim       = [ -60 20 ];  % latitude limits

        % Delay-embedding/finite-difference parameters
        % Indo-Pacific precip source data
        In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
        In.Src( 1 ).nXB       = 1;          % samples before main interval
        In.Src( 1 ).nXA       = 0;          % samples after main interval
        In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap';  % storage format 

        % Indo-Pacific precip target data
        In.Trg( 1 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;          % before main interval
        In.Trg( 1 ).nXA       = 0;          % samples after main interval
        In.Trg( 1 ).fdOrder   = 0;          % finite-difference order 
        In.Trg( 1 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap';  % storage format

        % Batches to partition the in-sample data
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
        In.nPhiKoopman    = numel( In.idxPhiKoopman );  % Koopman eigenfunctions to compute

    % SPCZ analysis using Indo-Pacific precipitation data
    case 'PacPrecip' 
        % In-sample data specification 

        % Indo-Pacific precip (source)
        In.Src( 1 ).field      = 'prate';     % physical field
        In.Src( 1 ).xLim       = [ 135 270 ];  % longitude limits
        In.Src( 1 ).yLim       = [ -35  35 ]; % latitude limits

        % Indo-Pacifid precip anomalies (target) 
        In.Trg( 1 ).field      = 'prate';  % physical field
        In.Trg( 1 ).xLim       = [ 28 290 ];   % longitude limits
        In.Trg( 1 ).yLim       = [ -60 20 ];  % latitude limits

        % Delay-embedding/finite-difference parameters
        % Indo-Pacific precip source data
        In.Src( 1 ).idxE      = 1 : 48;     % delay-embedding indices 
        In.Src( 1 ).nXB       = 1;          % samples before main interval
        In.Src( 1 ).nXA       = 0;          % samples after main interval
        In.Src( 1 ).fdOrder   = 1;          % finite-difference order 
        In.Src( 1 ).fdType    = 'backward'; % finite-difference type
        In.Src( 1 ).embFormat = 'overlap';  % storage format 

        % Indo-Pacific precip target data
        In.Trg( 1 ).idxE      = 1 : 1;      % delay embedding indices 
        In.Trg( 1 ).nXB       = 1;          % before main interval
        In.Trg( 1 ).nXA       = 0;          % samples after main interval
        In.Trg( 1 ).fdOrder   = 0;          % finite-difference order 
        In.Trg( 1 ).fdType    = 'backward'; % finite-difference type
        In.Trg( 1 ).embFormat = 'overlap';  % storage format

        % Batches to partition the in-sample data
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
        In.nPhiKoopman    = numel( In.idxPhiKoopman );  % Koopman eigenfunctions to compute


    otherwise
        error( 'Invalid experiment' )
    end

otherwise
    error( 'InValid dataset' )
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
