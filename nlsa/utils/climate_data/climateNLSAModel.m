function [model, In, Out] = climateNLSAModel(In, Out)
% CLIMATENLSAMODEL Low-level function to build NLSA models for climate datasets
%
%  [model, In] = climateNLSAModel(In) builds an NLSA model based on 
%  the model parameters specified in the structure In. 
%
%  [model, In, Out] = climateNLSAModel(In, Out) builds an NLSA model
%  with support for out-of-sample (test) data. The model parameters for the
%  in-sample (training) and out-of-sample (test) data are specified in the 
%  structures In and Out, respectively. 
%
%  climateNLSAModel uses the parameter values in In and Out to create arrays of 
%  nlsaComponent objects representing the in-sample and out-of-sample data, 
%  respectively. Then, it passses these arrays along with the parameter values 
%  to function /utils/nlsaModelFromPars. The latter, prepares appropriate 
%  arguments for the nlsaModel class constructors, and then calls the 
%  constructors to build the model.
%
%  climateNLSAModel is meant to be called by higher-level functions tailored to 
%  specific data analysis/forecasting experiments. 
%
%  For additional information see:
%
%      /nlsa/utils/nlsaModelFromPars.m
%      /nlsa/classes/nlsaModel_base/nlsaModel_base.m
%      /nlsa/classes/nlsaModel/nlsaModel.m
%      /nlsa/classes/nlsaModel_den/nlsaModel_den.m
%      /nlsa/classes/nlsaModel_den_ose/nlsaModel_den_ose.m
%
%  Structure field Src represents different physical variables (e.g., SST) 
%  employed in the kernel definition
%
%  Structure field Res represents different realizations (ensemble members)
%
% Modified 2023/06/08
 
%% PRELIMINARY CHECKS
% Check number of input arguments, and if we are doing out-of-sample extension
if nargin == 1
    Out   = [];
    ifOse = false; 
elseif nargin == 2
    ifOse = true;
else 
    error('Invalid number of input arguments.')
end

% Check if we are using kernel density estimation
if isfield(In, 'denType')
    ifDen = true;
else
    ifDen = false;
end

% Check that required high-level fields are present
if ~isfield(In, 'Src')
    error('Source field Src missing from in-sample parameters.')
end
if ~isfield(In, 'Trg')
    In.Trg = In.Src;
end
if ~isfield(In, 'Res')
    error('Realization field Res missing from in-sample parameters.')
end
if ifOse && ~isfield(Out, 'Res')
    warning(['Realization field Res missing from  out-of-sample ' ...
               'parameters. Reverting to default from in-sample data.'])
    Out.Res = In.Res;
end
if ifOse && isfield(Out, 'Trg')
    ifOutTrg = true; % Include out-of-sample target data
else
    ifOutTrg = false;
end

%% ROOT DIRECTORY NAMES
% In-sample data
if isfield(In, 'dataPath')
    inPath = In.dataPath;
else
    inPath      = fullfile(pwd, 'data/raw'); 
    In.dataPath = inPath;
end

% Out-of-sample data
if isfield(Out, 'dataPath')
    outPath = Out.dataPath;
else
    outPath      = fullfile(pwd, 'data/raw');
    Out.dataPath = outPath;
end


%% DELAY-EMBEDDING ORIGINGS
In.nC  = numel(In.Src); % number of source components
In.nCT = numel(In.Trg); % number of target compoents

% Maximum number of delay embedding lags, sample left out in the 
% beginning/end of the analysis interval for source data
In.nE = In.Src(1).idxE(end); 
In.nXB = In.Src(1).nXB; 
In.nXA = In.Src(1).nXA;
for iC = 2 : In.nC
    In.nE = max(In.nE, In.Src(iC).idxE(end));
    In.nXB = max(In.nXB, In.Src(iC).nXB);
    In.nXA = max(In.nXA, In.Src(iC).nXA);
end

% Maximum number of delay embedding lags, sample left out in the 
% beginning/end of the analysis interval for targe data
nETMin  = In.Trg(1).idxE(end); % minimum number of delays for target data
In.nET  = In.Trg(1).idxE(end); % maximum number of delays for target data
In.nXBT = In.Trg(1).nXB;
In.nXAT = In.Trg(1).nXA;
for iC = 2 : In.nCT
    In.nET = max(In.nET, In.Trg(iC).idxE(end));
    nETMin = min(In.nET, In.Trg(iC).idxE(end));
    In.nXBT = max(In.nXBT, In.Trg(iC).nXB);
    In.nXAT = max(In.nXAT, In.Trg(iC).nXA);
end
nEMax = max(In.nE, In.nET);
nXBMax = max(In.nXB, In.nXBT);
nXAMax = max(In.nXA, In.nXAT);

%% NUMBER OF STAMPLES FOR IN-SAMPLE DATA
In.nR  = numel(In.Res); % number of realizations, in-sample data
% Determine number of samples for in-sample data.
for iR = In.nR : -1 : 1
    % In.Res(iR).tNum:      timestemps (e.g., Matlab serial date numbers)
    % In.Res(iR).nS:    number of samples
    % In.Res(iR).idxT1: time origin for delay embedding
    % In.Res(iR).nSE:   number of samples after delay embedding
    % In.Res(iR).nSRec: number of samples for reconstruction
    In.Res(iR).nS = numel(In.Res(iR).tNum); 
    In.Res(iR).idxT1 = nEMax + nXBMax;      
    In.Res(iR).nSE = In.Res(iR).nS - In.Res(iR).idxT1 + 1 - nXAMax; 
    In.Res(iR).nSRec = In.Res(iR).nSE + nETMin - 1; 
end

%% OUT-OF-SAMPLE PARAMETER VALUES INHERITED FROM IN-SAMPLE DATA
if ifOse
    Out.tFormat      = In.tFormat; 
    Out.nC           = In.nC;  % number of source components
    Out.Src          = In.Src; % source component specification
    Out.nE           = In.nE;  % number of delays for source data
    Out.nET          = In.nET; % number of delays for target data
    Out.nXB          = In.nXB; % left-out source samples before main interval
    Out.nXA          = In.nXA; % left-out source samples after main interval
    Out.nXBT         = In.nXBT; % left-out target samples before main interval
    Out.nXAT         = In.nXAT; % left-out target samples after main interval 
    Out.lDist        = In.lDist; % local distance function
    Out.tol          = In.tol; % cone kernel tolerance
    Out.zeta         = In.zeta; % cone kernel parameter zeta 
    Out.coneAlpha    = In.coneAlpha; % cone kernel parameter alpha 
    Out.diffOpType   = In.diffOpType; % diffusion operator type
    Out.alpha        = In.alpha; % diffusion maps parameter alpha
    Out.nN           = In.nN; % nearest neighbors for OSE pairwise distance 
    Out.nNO          = Out.nN; % nearest neighbors for OSE diffusion operator
    Out.epsilon      = 1; % Bandwidth parameter
    Out.nPhi         = In.nPhi; % diffusion eigenfunctions to compute
    Out.nNO          = In.nN; % number of nearest neighbors for OSE 
    Out.idxPhiRecOSE = In.idxPhiRec; % eigenfunctions to reconstruct
end

if ifOse && ifOutTrg
    Out.nCT          = In.nCT; % number of target components
    Out.Trg          = In.Trg; % target component specification
end

if ifOse && ifDen
    Out.denType      = In.denType; % density estimation type
    Out.denLDist     = In.denLDist; % local distance for density estimation
    Out.denZeta      = In.denZeta; % cone kernel parameter zeta
    Out.denConeAlpha = In.denConeAlpha; % cone kernel paramter alpha
    Out.denNN        = In.denNN; % nearest neighbors for KDE
    Out.denND        = In.denND; % manifold dimension for density estimation
    Out.denEpsilon   = In.denEpsilon; % bandwidth parmeter for KDE
end


%% NUMBER OF SAMPLES AND TIMESTAMPS FOR OUT-OF-SAMPLE DATA
if ifOse
    Out.nR  = numel(Out.Res); % number of realizations, out-of-sample data
    % Determine number of samples for out-of-sample data.
    for iR = Out.nR : -1 : 1
        % Out.Res(iR).tNum:  timestemps (e.g., Matlab serial date numbers)
        % Out.Res(iR).nS:    number of samples
        % Out.Res(iR).idxT1: time origin for delay embedding
        % Out.Res(iR).nSE:   number of samples after delay embedding
        % Out.Res(iR).nSRec: number of samples for reconstruction
        Out.Res(iR).nS = numel(Out.Res(iR).tNum);
        Out.Res(iR).idxT1 = nEMax + nXBMax; 
        Out.Res(iR).nSE = Out.Res(iR).nS - Out.Res(iR).idxT1 + 1-nXAMax; 
        Out.Res(iR).nSRec = Out.Res(iR).nSE + nETMin - 1; 
    end
end

%% IN-SAMPLE DATA COMPONENTS
fList = nlsaFilelist('file', 'dataX.mat'); % filename for source data

% Loop over realizations for in-sample data
for iR = In.nR : -1 : 1

    tStr = [In.Res(iR).tLim{1} '-' In.Res(iR).tLim{2}]; 
    tagR = [In.Res(iR).experiment '_' tStr];
                                    
    % Source data assumed to be stored in a single batch
    partition = nlsaPartition('nSample', In.Res(iR).nS); 

    % Loop over source components
    for iC = In.nC : -1 : 1

        xyStr = sprintf('x%i-%i_y%i-%i', In.Src(iC).xLim(1), ...
                                          In.Src(iC).xLim(2), ...
                                          In.Src(iC).yLim(1), ...
                                          In.Src(iC).yLim(2));

        pathC = fullfile(inPath,  ...
                          In.Res(iR).experiment, ...
                          In.Src(iC).field,  ...
                          [xyStr '_' tStr]);
                                                   
        tagC = [In.Src(iC).field '_' xyStr];

        load(fullfile(pathC, 'dataGrid.mat'), 'nD')
        
        srcComponent(iC, iR) = nlsaComponent(...
                                    'partition',      partition, ...
                                    'dimension',      nD, ...
                                    'path',           pathC, ...
                                    'file',           fList, ...
                                    'componentTag',   tagC, ...
                                    'realizationTag', tagR );

    end

    % Loop over target components 
    for iC = In.nCT : -1 : 1

        xyStr = sprintf('x%i-%i_y%i-%i', In.Trg(iC).xLim(1), ...
                                          In.Trg(iC).xLim(2), ...
                                          In.Trg(iC).yLim(1), ...
                                          In.Trg(iC).yLim(2));

        pathC = fullfile(inPath,  ...
                          In.Res(iR).experiment, ...
                          In.Trg(iC).field,  ...
                          [xyStr '_' tStr]);
                                                   
        tagC = [In.Trg(iC).field '_' xyStr];

        load(fullfile(pathC, 'dataGrid.mat'), 'nD' )

        trgComponent(iC, iR) = nlsaComponent(...
                                    'partition',      partition, ...
                                    'dimension',      nD, ...
                                    'path',           pathC, ...
                                    'file',           fList, ...
                                    'componentTag',   tagC, ...
                                    'realizationTag', tagR );
    end

end

%% OUT-OF-SAMPLE DATA COMPONENTS 
if ifOse
    fList = nlsaFilelist('file', 'dataX.mat'); % filename for source data
    for iR = Out.nR : -1 : 1

        tStr = [Out.Res(iR).tLim{1} '-' Out.Res(iR).tLim{2}];
        tagR = [Out.Res(1).experiment '_' tStr];

        % Source data assumed to be stored in a single batch
        partition = nlsaPartition('nSample', Out.Res(iR).nS); 

        % Loop over out-of-sample source components
        for iC = Out.nC : -1 : 1

            xyStr = sprintf('x%i-%i_y%i-%i', Out.Src(iC).xLim(1), ...
                                              Out.Src(iC).xLim(2), ...
                                              Out.Src(iC).yLim(1), ...
                                              Out.Src(iC).yLim(2));

            pathC = fullfile(outPath,  ...
                              Out.Res(iR).experiment, ...
                              Out.Src(iC).field,  ...
                              [xyStr '_' tStr]);

            tagC = [Out.Src(iC).field '_' xyStr];

            % number of gridpoints
            load(fullfile(pathC, 'dataGrid.mat'), 'nD') 

            outComponent(iC, iR) = nlsaComponent(...
                                        'partition',      partition, ...
                                        'dimension',      nD, ...
                                        'path',           pathC, ...
                                        'file',           fList, ...
                                        'componentTag',   tagC, ...
                                        'realizationTag', tagR );
        end

        if ifOutTrg

            % Loop over out-of-sample target components
            for iC = Out.nCT : -1 : 1

                xyStr = sprintf('x%i-%i_y%i-%i', Out.Trg(iC).xLim(1), ...
                                                  Out.Trg(iC).xLim(2), ...
                                                  Out.Trg(iC).yLim(1), ...
                                                  Out.Trg(iC).yLim(2));

                pathC = fullfile(outPath,  ...
                                  Out.Res(iR).experiment, ...
                                  Out.Trg(iC).field,  ...
                                  [xyStr '_' tStr]);

                tagC = [Out.Trg(iC).field '_' xyStr];

                % number of gridpoints
                load(fullfile(pathC, 'dataGrid.mat'), 'nD') 

                outTrgComponent(iC, iR) = nlsaComponent(...
                                            'partition',      partition, ...
                                            'dimension',      nD, ...
                                            'path',           pathC, ...
                                            'file',           fList, ...
                                            'componentTag',   tagC, ...
                                            'realizationTag', tagR );
            end
        end
    end
end

%% ASSEMBLE DATA COMPONENTS AND CALL FUNCTION TO BUILD THE MODEL
Data.src = srcComponent;
Data.trg = trgComponent;
if ifOse
    Data.out = outComponent;
end
if ifOutTrg
    Data.outTrg = outTrgComponent;
end
Pars.In = In;
if ifOse
    Pars.Out = Out;
end

model = nlsaModelFromPars(Data, Pars);
