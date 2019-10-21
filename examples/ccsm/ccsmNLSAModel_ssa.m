function [ model, In ] = ccsmNLSAModel_ssa( experiment, idxPhiRec )
%% CCSMNLSAMODEL Build SSA model for  monthly data from the CCSM/CESM
%  climate models. 
% 
%  In is a data structure containing the model parameters (named after 
%  "in-sample," as opposed to "out-of-sample" data).
%
%  See script ccsmImport.m for additional details on the dynamical system.
%
%  For additional information on the arguments of nlsaMode_ssal( ... ) see 
%
%      ../classes/nlsaModel_base/parseTemplates.m
%      ../classes/nlsaModel_ssa/parseTemplates.m
%
% Modidied 2016/06/17

if nargin == 1
    idxPhiRec = 1;
end

switch experiment

    % NORTH PACIFIC SST
    case 'np_sst'

        % In-sample dataset parameters 
        In.tFormat             = 'yyyymm';    % time format
        In.Res( 1 ).yrLim      = [ 1 1300 ];  % time limits (in years) for realization 1 
        In.Res( 1 ).experiment = 'b40.1850';  % CCSM4/CESM experiment
        In.Src( 1 ).field      = 'SSTA';      % field for source component 1
        In.Src( 1 ).xLim       = [ 120 250 ]; % longitude limits
        In.Src( 1 ).yLim       = [ 20  65  ]; % latitude limits
        In.Trg( 1 ).field      = 'SSTA';      % field for target component 1
        In.Trg( 1 ).xLim       = [ 120 250 ]; % longitude limits
        In.Trg( 1 ).yLim       = [ 20  65  ]; % latitude limits

        % NLSA parameters
        In.Src( 1 ).idxE    = 1:3; %1 : 24;      % delay embedding indices for source component 1 
        In.Src( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        In.Trg( 1 ).idxE    = 1 : 24;   % delay embedding indices for target component 1
        In.Trg( 1 ).embFormat = 'overlap'; % storage format for delay embedding

        In.Res( 1 ).nXB = 0;         % samples to leave out before main interval
        In.Res( 1 ).nXA = 0;         % samples to leave out after main interval
        In.Res( 1 ).nB  = 64;        % batches to partition the in-sample data (realization 1)
        In.Res( 1 ).nBRec = In.Res( 1 ).nB; % batches for reconstructed data
        In.covOpType    = 'gl';       % covariance operator type
        In.nPhi         = 51;         % covariance eigenfunctions to compute
        In.nPhiPrj      = 51;         % eigenfunctions to project the data
        In.idxPhiRec    = idxPhiRec;  % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 33;     % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;      % SVD termporal patterns for reconstruction

    % INDO-PACIFIC SST
    case 'ip_sst'

        % In-sample dataset parameters 
        In.tFormat             = 'yyyymm';    % time format
        In.Res( 1 ).yrLim      = [ 1 1300 ];  % time limits (in years) for realization 1 
        In.Res( 1 ).experiment = 'b40.1850';  % CCSM4/CESM experiment
        In.Src( 1 ).field      = 'SSTA';       % field for source component 1
        In.Src( 1 ).xLim       = [ 28 290 ]; % longitude limits
        In.Src( 1 ).yLim       = [ -60 20 ]; % latitude limits
        In.Trg( 1 ).field      = 'SSTA';       % field for target component 1
        In.Trg( 1 ).xLim       = [ 28 290 ]; % longitude limits
        In.Trg( 1 ).yLim       = [ -60 20  ]; % latitude limits

        % NLSA parameters
        In.Src( 1 ).idxE    = 1 : 240;      % delay embedding indices for source component 1 
        In.Src( 1 ).embFormat = 'overlap'; % storage format for delay embedding
        In.Trg( 1 ).idxE    = 1 : 240;   % delay embedding indices for target component 1
        In.Trg( 1 ).embFormat = 'overlap'; % storage format for delay embedding

        In.Res( 1 ).nXB = 0;         % samples to leave out before main interval
        In.Res( 1 ).nXA = 0;         % samples to leave out after main interval
        In.Res( 1 ).nB  = 128;        % batches to partition the in-sample data (realization 1)
        In.Res( 1 ).nBRec = In.Res( 1 ).nB; % batches for reconstructed data
        In.covOpType    = 'gl';       % covariance operator type
        In.nPhi           = 201;        % covariance eigenfunctions to compute
        In.nPhiPrj      = In.nPhi;    % eigenfunctions to project the data
        In.idxPhiRec    = idxPhiRec;      % eigenfunctions for reconstruction
        In.idxPhiSVD    = 1 : 33;     % eigenfunctions for linear mapping
        In.idxVTRec     = 1 : 5;      % SVD termporal patterns for reconstruction


end

%% SSA MODEL

nlsaPath = './data/nlsa';

%==============================================================================
% Determine total number of samples, time origin for delay embedding

In.nC      = numel( In.Src );    % number of source components
In.nCT     = numel( In.Trg ); % number of target compoents
In.nR      = numel( In.Res ); % number of realizations

for iR = In.nR : -1 : 1
    In.Res( iR ).nYr  = In.Res( iR ).yrLim( 2 ) - In.Res( iR ).yrLim( 1 ) + 1; % number of years
    In.Res( iR ).nS   = 12 * In.Res( iR ).nYr; % number of monthly samples
    In.Res( iR ).tNum = zeros( 1, In.Res( iR ).nS );     % standard Matlab timestamps
    iS   = 1;
    for iYr = 1 : In.Res( iR ).nYr
        for iM = 1 : 12
            In.Res( iR ).tNum( iS ) = datenum( sprintf( '%04i%02i', In.Res( iR ).yrLim( 1 ) + iYr - 1, iM  ), 'yyyymm' );
            iS         = iS + 1;
        end
    end
end

In.nE = In.Src( 1 ).idxE( end ); % maximum number of delay embedding lags for source data
for iC = 2 : In.nC
    In.nE = max( In.nE, In.Src( iC ).idxE( end ) );
end
In.nET  = In.Trg( 1 ).idxE( end ); % maximum number of delay embedding lags for target data
for iC = 2 : In.nCT
    In.nET = max( In.nET, In.Trg( iC ).idxE( end ) );
end
In.idxT1   = max( In.nE, In.nET ) + In.Res.nXB;     % time origin for delay embedding
In.Res.nSE = In.Res.nS - In.idxT1 + 1 - In.Res.nXA; % sample number after embedding

%==============================================================================
% Setup nlsaComponent and nlsaProjectedComponent objects 

fList = nlsaFilelist( 'file', 'dataX.mat' ); % filename for source data

% Loop over realizations
for iR = In.nR : -1 : 1

    yr = sprintf( 'yr%i-%i', In.Res( iR ).yrLim( 1 ), In.Res( iR ).yrLim( 2 ) );
    tagR = [ In.Res( 1 ).experiment '_' yr ];
    partition = nlsaPartition( 'nSample', In.Res( iR ).nS ); % source data assumed to be stored in a single batch
    embPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSE, ...
                                        'nBatch',  In.Res( iR ).nB  );

    % Loop over source components
    for iC = In.nC : -1 : 1

        xy = sprintf( 'x%i-%i_y%i-%i', In.Src( iC ).xLim( 1 ), ...
                                       In.Src( iC ).xLim( 2 ), ...
                                       In.Src( iC ).yLim( 1 ), ...
                                       In.Src( iC ).yLim( 2 ) );

        pathC = fullfile( './data/raw/',  ...
                         In.Res( iR ).experiment, ...
                         In.Src( iC ).field,  ...
                         [ xy '_' yr ] );
                                                   
        tagC = [ In.Src( iC ).field '_' xy ];

        load( fullfile( pathC, 'dataGrid.mat' ), 'nD' ) % data space dimension
        In.Src( iC ).nD = nD;

        srcComponent( iC, iR ) = nlsaComponent( 'partition',      partition, ...
                                                'dimension',      In.Src( iC ).nD, ...
                                                'path',           pathC, ...
                                                'file',           fList, ...
                                                'componentTag',   tagC, ...
                                                'realizationTag', tagR  );

        switch In.Src( iC ).embFormat
            case 'evector'
                embComponent( iC, iR )= nlsaEmbeddedComponent_e( ...
                                        'idxE',    In.Src( iC ).idxE, ... 
                                        'nXB',     In.Res( iR ).nXB, ...
                                        'nXA',     In.Res( iR ).nXA );
            case 'overlap'
                embComponent( iC, iR) = nlsaEmbeddedComponent_o( ...
                                        'idxE',    In.Src( iC ).idxE, ...
                                        'nXB',     In.Res( iR ).nXB, ...
                                        'nXA',     In.Res( iR ).nXA );
        end
    
    end

    % Loop over target components
    for iC = In.nCT : -1 : 1

        xy = sprintf( 'x%i-%i_y%i-%i', In.Trg( iC ).xLim( 1 ), ...
                                       In.Trg( iC ).xLim( 2 ), ...
                                       In.Trg( iC ).yLim( 1 ), ...
                                       In.Trg( iC ).yLim( 2 ) );

        pathC = fullfile( './data/raw/',  ...
                         In.Res( iR ).experiment, ...
                         In.Trg( iC ).field,  ...
                         [ xy '_' yr ] );
                                                   
        tagC = [ In.Trg( iC ).field '_' xy ];

        load( fullfile( pathC, 'dataGrid.mat' ), 'nD' ) % data space dimension
        In.Trg( iC ).nD = nD;

        trgComponent( iC, iR ) = nlsaComponent( 'partition',      partition, ...
                                                'dimension',      In.Trg( iC ).nD, ...
                                                'path',           pathC, ...
                                                'file',           fList, ...
                                                'componentTag',   tagC, ...
                                                'realizationTag', tagR  );

        switch In.Trg( iC ).embFormat
            case 'evector'
                trgEmbComponent( iC, iR )= nlsaEmbeddedComponent_e( ...
                                      'idxE',    In.Trg( iC ).idxE, ... 
                                      'nXB',     In.Res( iR ).nXB, ...
                                      'nXA',     In.Res( iR ).nXA );
            case 'overlap'
                trgEmbComponent( iC, iR) = nlsaEmbeddedComponent_o( ...
                                           'idxE',    In.Trg( iC ).idxE, ...
                                           'nXB',     In.Res( iR ).nXB, ...
                                           'nXA',     In.Res( iR ).nXA );
        end


        prjComponent( iC ) = nlsaProjectedComponent( ...
                                 'nBasisFunction', In.nPhiPrj );
    end
end


%==============================================================================
% Covariance operators 

switch In.covOpType
    % global storage format
    case 'gl'
        covOp = nlsaCovarianceOperator_gl( 'nEigenfunction', In.nPhi );  
end

%==============================================================================
% Linear map for SVD of the target data 
linMap = nlsaLinearMap_gl( 'basisFunctionIdx', In.idxPhiSVD );

%==============================================================================
% Reconstructed components
In.Res.nSRec  = In.Res.nSE + In.nE - 1; % number of reconstructed samples
for iR = In.nR : -1 : 1

    % Partition for reconstructed data
    recPartition( iR ) = nlsaPartition( 'nSample', In.Res( iR ).nSRec, ... 
                                        'nBatch',  In.Res( iR ).nBRec );

    % Reconstructed data from diffusion eigenfnunctions
    recComponent( 1, iR ) = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxPhiRec );

    % Reconstructed data from SVD 
    svdRecComponent( 1, iR ) = nlsaComponent_rec_phi( 'basisFunctionIdx', In.idxVTRec );
end

%==============================================================================
% Build NLSA model    
model = nlsaModel_ssa( 'path',                            nlsaPath, ...
                       'timeFormat',                      In.tFormat, ...
                       'sourceTime',                      { In.Res.tNum }, ...
                       'sourceComponent',                 srcComponent, ...
                       'embeddingOrigin',                 In.idxT1, ...
                       'embeddingTemplate',               embComponent, ...
                       'embeddingPartition',              embPartition, ...
                       'covarianceOperatorTemplate',      covOp, ...
                       'targetComponent',                 trgComponent, ...
                       'targetEmbeddingTemplate',         trgEmbComponent, ...
                       'projectionTemplate',              prjComponent, ...
                       'reconstructionTemplate',          recComponent, ...
                       'reconstructionPartition',         recPartition, ...
                       'linearMapTemplate',               linMap, ...
                       'svdReconstructionTemplate',       svdRecComponent );

