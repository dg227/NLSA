function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel_err class
%   constructor from templates.
%
%   The arguments of parseTemplates are passed as property name-property value
%   pairs using the syntax:
%
%   propNameVals  = parseTemplates( propName1, propVal1, propName2, propVal2, ... ).
%
%   The following properties can be specified in addition to those available in the
%   parseTemplates method of the nlsaModel_ose superclass:

%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2014/12/15    


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel_err.listConstructorProperties;         
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );

%% SUPERCLASS CONSTRUCTOR ARGUMENTS

parentConstrArgs = nlsaModel_ose.parseTemplates( varargin{ : } );
iOutTime = [];
for iProp = 1 : 2 : numel( parentConstrArgs )
    switch parentConstrArgs{ iProp }
        case 'embComponent'
            iEmbComponent = iProp + 1;
        case 'trgEmbComponent'
            iTrgEmbComponent = iProp + 1;
        case 'prjComponent'
            iPrjComponent = iProp + 1;
        case 'outTime'
            iOutTime = iProp + 1;
        case 'outComponent'
            iOutComponent = iProp + 1;
        case 'outEmbComponent'
            iOutEmbComponent = iProp + 1;
        case 'oseEmbComponent'
            iOseEmbComponent = iProp + 1;
        case 'osePairwiseDistance' 
            iOsePDist = iProp + 1;
        case 'diffusionOperator'
            iDiffOp = iProp + 1;
        case 'oseDiffusionOperator'  
            iOseDiffOp = iProp + 1;
    end
end 
[ nCT, nR ] = size( parentConstrArgs{ iTrgEmbComponent } );
[ ~, nRO ]  = size( parentConstrArgs{ iOseEmbComponent } );


%% MODEL PATH
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'path' )
        if isSet == 1
            error( 'Path argument has been already specified' )
        end
        if ischar( varargin{ i + 1 } )
            modelPath = varargin{ i + 1 };
            isSet = 1;
        else
            error( 'Invalid path specification' )
        end
    end
end
if ~isSet
    modelPath = pwd;
end

%% OS TARGET TIMESTAMPS
% Set target time data
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outTrgTime' )
        iOutTrgTime = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTrgTime' )                   
        if ~isempty( propVal{ iOutTrgTime } )
            error( 'Source time data have been already specified' )
        end
        propVal{ iOutTrgTime } = varargin{ i + 1 };
        ifProp( iOutTrgTime )  = true;
    end
end
if isempty( propVal{ iOutTrgTime } ) && ~isempty( iOutTime )
    propVal{ iOutTrgTime } = propVal{ iOutTime };
    ifProp( iOutTrgTime ) = true;
end

           
%% OS TARGET DATA

% Import target data and determine the number of samples and dimension 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outTrgComponent' )
        iOutTrgComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTargetComponent' )
        if ~isempty( propVal{ iOutTrgComponent } )
            error( 'Target components have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaComponent' )
            error( 'OSE target data must be specified as an array of nlsaComponent objects' )
        end
        propVal{ iOutTrgComponent } = varargin{ i + 1 };
        ifProp( iOutTrgComponent )  = true;
    end
end
if isempty( propVal{ iOutTrgComponent } )
    propVal{ iOutTrgComponent } = parentConstrArgs{ iOutComponent };
end     
nCT = size( propVal{ iOutTrgComponent }, 1 );
if size( propVal{ iOutTrgComponent }, 2 ) ~= nRO
    error( 'The number of OSE source and target realizations must be equal' )
end


%% OS TARGET EMBEDDED DATA
% Parse embedding templates for the target data   
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outTrgEmbComponent' )
        iOutTrgEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTargetEmbeddingTemplate' )
        if ~isempty( propVal{ iOutTrgEmbComponent } ) 
            error( 'Time-lagged embedding templates for the target data have been already specified' )
        end
        if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
            && isscalar( varargin{ i + 1 } )
            propVal{ iOutTrgEmbComponent } = repmat( varargin{ i + 1 }, [ nCT 1 ] );
        elseif    isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nCT
            propVal{ iOutTrgEmbComponent } = varargin{ i + 1 };
            if size( propVal{ iOutTrgEmbComponent }, 2 ) > 1 
                propVal{ iOutTrgEmbComponent } = propVal{ iOutTrgEmbComponent }';
            end
        else
            error( 'Time-lagged embedding templates must be specified as a scalar nlsaEmbeddedComponent object, or vector of nlsaEmbeddedComponent objects of size equal to the number of components' )
        end
        ifProp( iOutTrgEmbComponent ) = true;
    end
end
if isempty( propVal{ iOutTrgEmbComponent } )
    propVal{ iOutTrgEmbComponent } = parentConstrArgs{ iOutEmbComponent };
    ifProp( iOutTrgEmbComponent )  = true;
else
    for iC = 1 : nCT
        propVal{ iOutTrgEmbComponent }( iC ) = setDimension( propVal{ iOutTrgEmbComponent }( iC ), ...
                                            getDimension( propVal{ iOutTrgComponent }( iC ) ) );
    end
    
    % Determine time limits for embedding origin 
    minEmbeddingOrigin = getMinOrigin( propVal{ iOutTrgEmbComponent } );

    % Replicate template to form target embedded component array
    propVal{ iOutTrgEmbComponent } = repmat( propVal{ iOutTrgEmbComponent }, [ 1 nRO ] );             
    
    % Parse embedding origin templates
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outTargetEmbeddingOrigin' )
            if isSet
                error( 'Time-lagged embedding origins for the target data have been already specified' )
            end
            if ispsi( varargin{ i + 1 } )
                embeddingOrigin = repmat( varargin{ i + 1 }, [ 1 nRO ] );
            elseif isvector( varargin{ i + 1 } ) && numel( varargin{ i + 1 } ) == nRO 
                embeddingOrigin = varargin{ i + 1 };
            end
            isSet = true;
        end
    end
    if ~isSet
        embeddingOrigin = minEmbeddingOrigin * ones( 1, nRO );
    end
    for iR = 1 : nRO
        if embeddingOrigin( iR ) < minEmbeddingOrigin
            error( 'Time-lagged embedding origin is below minimum value' )
        end
        for iC = 1 : nCT
            propVal{ iOutTrgEmbComponent }( iC, iR ) = setOrigin( propVal{ iOutTrgEmbComponent }( iC, iR ), embeddingOrigin( iR ) );
        end
    end

    % Determine maximum number of samples in each embedded component
    maxNSRET = zeros( 1, nRO );
    for iR = 1 : nRO
        maxNSRET( iR ) = getMaxNSample( propVal{ iOutTrgEmbComponent }( :, iR ), ...
                                        getNSample( parentConstrArgs{ iOutTrgComponent }( :, iR ) ) );
    end

    % Parse partition templates for the target data
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outTargetPartitionTemplate' )
            if isSet
                error( 'Partition templates for the target data have already been specified' )
            end
            if isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isscalar( varargin{ i + 1 } )
                partition = repmat( varargin{ i + 1 }, [ 1 nRO ] );
            elseif isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nRO
                partition = varargin{ i + 1 };
            else
                error( 'Data partitions must be specified as a scalar nlsaPartition object, or a vector of nlsaPartition objects of size equal to the number of enemble realizations' )
            end
            isSet = true;
        end
    end 
    if any( getNSample( partition ) ~= nSRE )
        error( 'Incompatible number of target samples' )
    end

    for iR = 1 : nRO
        if getNSample( partition( iR ) ) > maxNSRET( iR )
             msgStr = [ 'Number of time-lagged embedded samples ', ...
                        int2str( getNSample( partition( iR ) ) ), ...
                        ' is above maximum value ', ...
                        int2str( maxNSRE( iR ) ) ];
            error( msgStr ) 
        end
        for iC = 1 : nCT
            propVal{ iOutTrgEmbComponent }( iC, iR ) = setPartition( propVal{ iOutTrgEmbComponent }( iC, iR ), partition( iR ) );
        end 
    end
end
nDET = getEmbeddingSpaceDimension( propVal{ iOutTrgEmbComponent }( :, 1 ) ); % Needed later to set up linear maps

% Setup target embedded component tags, directories, and filenames
for iR = 1 : nRO
    for iC = 1 : nCT

        propVal{ iOutTrgEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iOutTrgEmbComponent }( iC, iR ) );
        propVal{ iOutTrgEmbComponent }( iC, iR ) = ...
            setComponentTag( ...
                propVal{ iOutTrgEmbComponent }( iC, iR ), ...
                getComponentTag( propVal{ iOutTrgComponent }( iC, 1 ) ) );
        propVal{ iOutTrgEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iOutTrgEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iOutTrgComponent }( 1, iR ) ) );

        tag  = getTag( propVal{ iOutTrgEmbComponent }( iC, iR ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );

        propVal{ iOutTrgEmbComponent }( iC, iR ) = ...
            setPath( propVal{ iOutTrgEmbComponent }( iC, iR ), pth );

        propVal{ iOutTrgEmbComponent }( iC, iR ) = ...
            setDefaultSubpath( propVal{ iOutTrgEmbComponent }( iC, iR ) );


        propVal{ iOutTrgEmbComponent }( iC, iR ) = ...
            setDefaultFile( propVal{ iOutTrgEmbComponent }( iC, iR ) );
    end
end
mkdir( propVal{ iOutTrgEmbComponent } )

% PAIRWISE DISTANCES FOR OS DATA
% Parse distance template and set distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outPairwiseDistance' )
        iOutPDist = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outPairwiseDistanceTemplate' )
        if ~isempty( propVal{ iOutPDist } )
            error( 'A model pairwise distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iOutPDist } = varargin{ i + 1 };
            ifProp( iOutPDist )  = true;
        else
            error( 'The model pairwise distance template must be specified as a scalar nlsaPairwiseDistance object' )
        end
    end
end
if isempty( propVal{ iOutPDist } )
    propVal{ iOutPDist } = nlsaPairwiseDistance( 'nearestNeighbors', round( nSE / 10 ) );
    ifProp( iOutPDist )  = true;
end
nN = getNNeighbors( propVal{ iOutPDist } );

propVal{ iOutPDist } = setPartition( propVal{ iOutPDist }, ...
                                         getPartition( parentConstrArgs{ iOsePDist } ) );
propVal{ iOutPDist } = setPartitionTest( propVal{ iOutPDist }, ...
                                             getPartition( parentConstrArgs{ iOsePDist } ) );
tag = getTag( propVal{ iOutPDist } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iOutPDist } = setTag( propVal{ iOutPDist }, ...
        [ tag getDefaultTag( propVal{ iOutPDist } ) ] ); 

% Determine distance-specific directories
pth = concatenateTags( parentConstrArgs{ iOseEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outComponentName' ) 
        if ~isSet
            pth{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'outComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
        end
    end  
end
pth = strjoin_e( pth, '_' );

% Assign pairwise distance paths and filenames
modelPathD = fullfile( modelPath, 'processed_data', pth, ...
                       getTag( propVal{ iOutPDist } )  );
propVal{ iOutPDist } = setPath( propVal{ iOutPDist }, modelPathD );
propVal{ iOutPDist } = setDefaultSubpath( propVal{ iOutPDist } ); 
propVal{ iOutPDist } = setDefaultFile( propVal{ iOutPDist } );
mkdir( propVal{ iOutPDist } )


%% SYMMETRIC DISTANCE FOR THE OS DATA
% Parse symmetric distance template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outSymmetricDistance' )
        iOutSDist = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outSymmetricDistanceTemplate' )
        if ~isempty( propVal{ iOutSDist } )
            error( 'A symmetric distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaSymmetricDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iOutSDist } = varargin{ i + 1 };
            ifProp( iOutSDist ) = true;
        else
            error( 'The model symmetric distance template must be specified as a scalar nlsaSymmetricDistance object' )
        end
    end
end

if isempty( propVal{ iOutSDist } )
    propVal{ iOutSDist } = nlsaSymmetricDistance_gl( 'nearestNeighbors', nN );
    ifProp( iOutSDist ) = true;
end

if getNNeighbors( propVal{ iOutSDist } ) > nN
    error( 'The number of nearest neighbors in the model symmetric distance matrix cannot exceed the number of neareast neighbors in the pairwise (non-symmetric) distance matrix' )
end
propVal{ iOutSDist } = setPartition( propVal{ iOutSDist }, ...
                               getPartition( propVal{ iOutPDist } ) );
tag = getTag( propVal{ iOutSDist } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iOutSDist } = setTag( propVal{ iOutSDist }, ...
        [ tag getDefaultTag( propVal{ iOutSDist } ) ] ); 

% Assign symmetric distance paths and filenames
modelPathS            = fullfile( modelPathD, getTag( propVal{ iOutSDist } ) );
propVal{ iOutSDist } = setPath( propVal{ iOutSDist }, modelPathS );
propVal{ iOutSDist } = setDefaultSubpath( propVal{ iOutSDist } );
propVal{ iOutSDist } = setDefaultFile( propVal{ iOutSDist } );
mkdir( propVal{ iOutSDist } )


%% DIFFUSION OPERATOR FOR THE OS DATA
% Parse diffusion operator template 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outDiffusionOperator' )
        iOutDiffOp = iProp;
        break
    end  
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outDiffusionOperatorTemplate' )
        if ~isempty( propVal{ iOutDiffOp } )
            error( 'A diffusion operator template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iOutDiffOp } = varargin{ i + 1 };
        else
            error( 'The model diffusion operator template must be specified as a scalar nlsaDiffusionOperator object' )
        end
        ifProp( iOutDiffOp ) = true;
    end
end
if isempty( propVal{ iOutDiffOp } )
    propVal{ iOutDiffOp } = nlsaDiffusionOperator();
    ifProp( iOutDiffOp ) = true;
end
propVal{ iOutDiffOp } = setPartition( propVal{ iOutDiffOp }, ...
                            getPartition( propVal{ iOutPDist } ) );
propVal{ iOutDiffOp } = setPartitionTest( propVal{ iOutDiffOp }, ...
                            getPartition( propVal{ iOutPDist } ) );
propVal{ iOutDiffOp } = setNNeighbors( propVal{ iOutDiffOp }, ...
                                    getNNeighborsMax( propVal{ iOutSDist } ) );
tag = getTag( propVal{ iOutDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iOutDiffOp } = setTag( propVal{ iOutDiffOp }, ...
        [ tag getDefaultTag( propVal{ iOutDiffOp } ) ] ); 

% Assign diffusion operator  paths and filenames
modelPathL = fullfile( modelPathS, getTag( propVal{ iOutDiffOp } ) );
propVal{ iOutDiffOp } = setDefaultSubpath( propVal{ iOutDiffOp } );
propVal{ iOutDiffOp } = setPath( propVal{ iOutDiffOp }, modelPathL );
mkdir( propVal{ iOutDiffOp } )
propVal{ iOutDiffOp } = setDefaultFile( propVal{ iOutDiffOp } );


%% OS PROJECTED DATA
% Parse projection templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outPrjComponent' )
        iOutPrjComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outProjectionTemplate' )
        if ~isempty( propVal{ iOutPrjComponent } )
            error( 'A projection template for the OS data has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
            && isscalar( varargin{ i + 1 } ) 
            propVal{ iOutPrjComponent } = repmat( varargin{ i + 1 }, [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaProjectedComponent' ) ...
           && isvector( varargin{ i + 1 } ) ...
           && numel( varargin{ i + 1 } ) == nCT
            propVal{ iOutPrjComponent } = varargin{ i + 1 };
            if isrow( propVal{ iOutPrjComponent } )
                propVal{ iOutPrjComponent } = propVal{ iOutPrjComponent }';
            end
        else
            error( 'The projection template must be specified as a scalar nlsaProjectedComponent object, or vector of nlsaProjectedComponent objects of size equal to the number of target components' )
        end
        ifProp( iOutPrjComponent ) = true;
    end
end
if isempty( propVal{ iOutPrjComponent } )
    for iC = nCT : -1 : 1
        propVal{ iOutPrjComponent }( iC ) = nlsaProjectedComponent( 'nBasisFunction', ...
          getNEigenfunction( propVal{ iOutDiffOp } ) );
    end
    ifProp( iOutPrjComponent ) = true;
end
partition = getPartition( parentConstrArgs{ iOutTrgEmbComponent }( 1, : ) );
for iC = 1 : nCT
    % Setup collective tags for projected data
    pth = concatenateTags( parentConstrArgs{ iOutTrgEmbComponent }( iC, : ) );
    pth = pth{ 1 };
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outProjectionComponentName' )
            if ~isSet
                pth = varargin{ i + 1 };
                break
            else
                error( 'Projection component name for the OS data has been already specified' )
            end      
        end  
    end
    propVal{ iOutPrjComponent }( iC ) = setPartition( ...
        propVal{ iOutPrjComponent }( iC ), partition );
    propVal{ iOutPrjComponent }( iC ) = setPath( propVal{ iOutPrjComponent }( iC ), ...
    fullfile( modelPathL, pth ) );
    propVal{ iOutPrjComponent }( iC ) = setEmbeddingSpaceDimension( propVal{ iOutPrjComponent }( iC ), ...
      getEmbeddingSpaceDimension( parentConstrArgs{ iOutTrgEmbComponent }( iC ) ) ); 
    propVal{ iOutPrjComponent }( iC ) = setDefaultSubpath( propVal{ iOutPrjComponent }( iC ) );
    propVal{ iOutPrjComponent }( iC ) = setDefaultFile( propVal{ iOutPrjComponent }( iC ) );
end
mkdir( propVal{ iOutPrjComponent } )

%% REFERENCE IN-SAMPLE RESTRICTION (ISR) DATA 
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'isrRefComponent' )
        iIsrRefComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'isrReferenceTemplate' )
        if ~isempty( propVal{ iIsrRefComponent } )
            error( 'Reference data templates for ISR error have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent_rec' ) ...
          || (  ~isscalar( varargin{ i + 1 } ) ...
             && ~( isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } == nCT ) ) )
            error( 'The reference data templates must be specified as a scalar nlsaEmbeddedComponent_rec object or a vector of nlsaEmbeddedComponent_rec objects with number of elements equal to the number of test components' )
        end

        if any( getMaxBasisFunctionIndex( varargin{ i + 1 } ) ...
                > getNBasisFunction( parentConstrArgs{ iPrjComponent } ) )
            error( 'Insufficient number of projected basis functions' )
        end

        for iR = nR : -1 : 1
            for iC = nCT : -1 : 1
                iCSet = min( iC, numel( varargin{ i + 1 } ) );
                propVal{ iIsrRefComponent }( iC, iR )  = ...
                    nlsaEmbeddedComponent_rec( ...
                        parentConstrArgs{ iTrgEmbComponent }( iC, iR ) );
                propVal{ iIsrRefComponent }( iC, iR ) = ...
                    setBasisFunctionIndices( ...
                        propVal{ iIsrRefComponent }, ...
                        getBasisFunctionIndices( ...
                            varargin{ i + 1 }( iCSet ) ) );
                propVal{ iIsrRefComponent }( iC, iR ) = ...
                    setPath( propVal{ iIsrRefComponent }( iC, iR ), ...
                             getPath( parentConstrArgs{ iPrjComponent }( iC, iR ) ) );
                propVal{ iIsrRefComponent }( iC, iR ) = ...
                    setDefaultTag( propVal{ iIsrRefComponent }( iC, iR ) );
                propVal{ iIsrRefComponent }( iC, iR ) = ...
                    setDefaultSubpath( propVal{ iIsrRefComponent }( iC, iR ) );
                propVal{ iIsrRefComponent }( iC, iR ) = setDefaultFile( propVal{ iIsrRefComponent }( iC, iR ) );

            end
        end
        mkdir( propVal{ iIsrRefComponent } )
        ifProp( iIsrRefComponent ) = true;
    end
end
if isempty( propVal{ iIsrRefComponent } )
    propVal{ iIsrRefComponent } = parentConstrArgs{ iTrgEmbComponent };
    ifProp( iIsrRefComponent ) = true;
end


%% REFERENCE OSE DATA
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'oseRefComponent' )
        iOseRefComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseReferenceTemplate' )
        if ~isempty( propVal{ iOseRefComponent } )
            error( 'Reference data templates for the OSE error have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent_rec' ) ...
          || ( ~isscalar( varargin{ i + 1 } ) ...
             && ~( isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } == nCT ) ) )
            error( 'The reference data templates must be specified as a scalar nlsaEmbeddedComponent_rec object or a vector of nlsaEmbeddedComponent_rec objects with number of elements equal to the number of test components' )
        end
        if any( getMaxBasisFunctionIndex( varargin{ i + 1 } ) ...
                > getNBasisFunction( propVal{ iOutPrjComponent } ) )
            error( 'Insufficient number of projected basis functions' )
        end
        for iR = nRO : -1 : 1
            for iC = nCT : -1 : 1
                iCSet = min( iC, numel( varargin{ i + 1 } ) );
                propVal{ iOseRefComponent }( iC, iR )  = ...
                    nlsaEmbeddedComponent_rec( ...
                        parentConstrArgs{ iOutTrgEmbComponent }( iC, iR ) );
                propVal{ iOseRefComponent }( iC, iR ) = ...
                    setBasisFunctionIndices( ...
                        propVal{ iOseRefComponent }, ...
                        getBasisFunctionIndices( ...
                            varargin{ i + 1 }( iCSet ) ) );
                propVal{ iOseRefComponent }( iC, iR ) = ...
                    setPath( propVal{ iOseRefComponent }( iC, iR ), ...
                              getPath( propVal{ iOutPrjComponent }( iC, iR ) ) );
                propVal{ iOseRefComponent }( iC, iR ) = ...
                    setDefaultTag( propVal{ iOseRefComponent }( iC, iR ) );
                propVal{ iOseRefComponent }( iC, iR ) = ...
                    setDefaultSubpath( propVal{ iOseRefComponent }( iC, iR ) );
                propVal{ iOseRefComponent }( iC, iR ) = ...
                    setDefaultFile( propVal{ iOseRefComponent }( iC, iR ) );
mkdir( propVal{ iOseRefComponent } )

               
            end
        end
        mkdir( propVal{ iOseRefComponent } )
        ifProp( iOseRefComponent ) = true;
    end
end
if isempty( propVal{ iOseRefComponent } )
    propVal{ iOseRefComponent } = parentConstrArgs{ iOutTrgEmbComponent };
    ifProp( iOseRefComponent ) = true;
end


%% PAIRWISE DISTANCES FOR ISR 
% Parse ISR distance template and set ISR distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'isrPairwiseDistance' )
        iIsrPDist = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'isrPairwiseDistanceTemplate' )
        if ~isempty( propVal{ iIsrPDist } )
            error( 'A pairwise distance template for the model data has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iIsrPDist } = varargin{ i + 1 };
            ifProp( iIsrPDist )  = true;
            ifRetainTag          = true;
        else
            error( 'The OSE pairwise distance template must be specified as a scalar nlsaPairwiseDistance object' )
        end
    end
end
if isempty( propVal{ iIsrPDist } )
    propVal{ iIsrPDist } = propVal{ iOutPDist };
    ifProp( iIsrPDist )  = true;
    ifRetainTag             = false;
end

% Partition for query (OSE) data
propVal{ iIsrPDist } = setPartition( propVal{ iIsrPDist }, ...
                                        getPartitionTest( parentConstrArgs{ iOsePDist } ) );

% partition for test (in-sample) data 
propVal{ iIsrPDist } = setPartitionTest( propVal{ iIsrPDist }, ...
                                         getPartition( parentConstrArgs{ iOsePDist } ) ); 

if ifRetainTag
    tag = getTag( propVal{ iIsrPDist } );
    if ~isempty( tag )
        tag = [ tag '_' ];
    end
else
    tag = [];
end
propVal{ iIsrPDist } = setTag( propVal{ iIsrPDist }, ...
        [ tag getDefaultTag( propVal{ iIsrPDist } ) ] ); 

% Determine distance-specific directories
pth = concatenateTags( parentConstrArgs{ iEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'srcComponentName' ) 
        if ~isSet
            pth{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'srcComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'srcRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
            break
        end
    end  
end
pth = strjoin_e( pth, '_' );

% Assign pairwise distance paths and filenames
modelPathDO = fullfile( getPath( propVal{ iOutPDist } ), ...
                        pth, ...
                        getTag( propVal{ iIsrPDist } ) ); 
propVal{ iIsrPDist } = setPath( propVal{ iIsrPDist }, modelPathDO ); 
propVal{ iIsrPDist } = setDefaultSubpath( propVal{ iIsrPDist } );
propVal{ iIsrPDist } = setDefaultFile( propVal{ iIsrPDist } );
mkdir( propVal{ iIsrPDist } )

%% ISR DIFFUSION OPERATOR
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'isrDiffusionOperator' )
        iIsrDiffOp = iProp;
        break
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'isrDiffusionOperatorTemplate' )
        if ~isSet
            if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator_ose' ) ...
               && isscalar( varargin{ i + 1 } )
                propVal{ iIsrDiffOp } = varargin{ i + 1 };
                isSet = true;
            else
                error( 'The ISR diffusion operator template must be specified as a scalar nlsaDiffusionOperator_ose object' )
            end
        else       
            error( 'An OSE diffusion operator template has been already specified' )
        end
    end
end
if ~isSet
    propVal{ iIsrDiffOp } = nlsaDiffusionOperator_ose( parentConstrArgs{ iDiffOp } );
end
ifProp( iIsrDiffOp )  = true;
propVal{ iIsrDiffOp } = setPartition( propVal{ iIsrDiffOp }, ...
                                    getPartition( propVal{ iIsrPDist } ) );
propVal{ iIsrDiffOp } = setPartitionTest( propVal{ iIsrDiffOp }, ...
                                    getPartitionTest( propVal{ iIsrPDist } ) );
tag = getTag( propVal{ iIsrDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iIsrDiffOp } = setTag( propVal{ iIsrDiffOp }, ...
        [ tag getDefaultTag( propVal{ iIsrDiffOp } ) ] ); 

% Assign diffusion operator paths and filenames
modelPathLO           = fullfile( getPath( propVal{ iOutDiffOp } ), ...
                                  pth, ...
                                  getTag( propVal{ iIsrPDist } ), ...
                                  getTag( propVal{ iIsrDiffOp } ) );
propVal{ iIsrDiffOp } = setDefaultSubpath( propVal{ iIsrDiffOp } );
propVal{ iIsrDiffOp } = setPath( propVal{ iIsrDiffOp }, modelPathLO );
mkdir( propVal{ iIsrDiffOp } )
propVal{ iIsrDiffOp } = setDefaultFile( propVal{ iIsrDiffOp } );


%% STATE AND VELOCITY ISR COMPONENTS
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'isrEmbComponent' )
        iIsrEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'isrTemplate' )
        if ~isempty( propVal{ iIsrEmbComponent } )
            error( 'OSE templates have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent_ose_n' ) ...
            || ( ~isscalar( varargin{ i + 1 } ) ...
                 && ~( isvector( varargin{ i + 1 } ) ...
                 && numel( varargin{ i + 1 } == nCT ) ) )
            error( 'The ISR templates must be specified as a scalar nlsaEmbeddedComponent_ose_n object or a vector of nlsaEmbeddedComponent_ose_n objects with number of elements equal to the number of test components' )
        end
        if any( getMaxEigenfunctionIndex( varargin{ i + 1 } ) ...
                > getNEigenfunction( propVal{ iIsrDiffOp } ) )
            error( 'Insufficient number of ISR diffusion eigenfunctions' )
        end
        for iC = nCT : -1 : 1
            iCSet = min( iC, numel( varargin{ i + 1 } ) );
            propVal{ iIsrEmbComponent }( iC, 1 )  = ...
                nlsaEmbeddedComponent_ose_n( ...
                    parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) );
            propVal{ iIsrEmbComponent }( iC, 1 ) = ...
                setEigenfunctionIndices( ...
                    propVal{ iIsrEmbComponent }( iC, 1 ), ...
                    getEigenfunctionIndices( ...
                        varargin{ i + 1 }( iCSet ) ) );
        end
        ifProp( iIsrEmbComponent ) = true;
    end
end
if isempty( propVal{ iIsrEmbComponent } )
    for iC = nCT : -1 : 1
        propVal{ iIsrEmbComponent }( iC, 1 ) = ...
            nlsaEmbeddedComponent_ose( ...
                parentConstrArgs{ iOseTrgEmbComponent }( iC, 1 ) );
    end
    ifProp( iIsrEmbComponent ) = true;
end
propVal{ iIsrEmbComponent } = repmat( propVal{ iIsrEmbComponent }, [ 1 nR ] );
for iR = 1 : nR
    for iC = 1 : nCT
        propVal{ iIsrEmbComponent }( iC, iR ) = ...
            setPartition( ...
                propVal{ iIsrEmbComponent }( iC, iR ), ...
                getPartition( parentConstrArgs{ iEmbComponent }( 1, iR ) ) );
        propVal{ iIsrEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iIsrEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iEmbComponent }( 1, iR ) ) );
        propVal{ iIsrEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iIsrEmbComponent }( iC, iR ) );
        
        propVal{ iIsrEmbComponent }( iC, iR ) = ...
            setPath( propVal{ iIsrEmbComponent }( iC, iR ), ...
                fullfile( ...
                    modelPathLO, ...
                    strjoin_e( getTag( propVal{ iIsrEmbComponent }( iC, iR ) ), '_' ) ) );
        propVal{ iIsrEmbComponent }( iC, iR ) = setDefaultSubpath( propVal{ iIsrEmbComponent }( iC, iR ) );
        propVal{ iIsrEmbComponent }( iC, iR ) = setDefaultFile( propVal{ iIsrEmbComponent }( iC, iR ) );
    end
end
mkdir( propVal{ iIsrEmbComponent } )

%% OSE STATE AND VELOCITY ERROR
% Determine error-specific directories
pth = concatenateTags( parentConstrArgs{ iOseEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTargetComponentName' ) 
        if ~isSet
            pth{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'outTargetComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTargetRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
        end
    end  
end
pth = strjoin_e( pth, '_' );
if isa( propVal{ iOseRefComponent }, 'nlsaEmbeddedComponent_rec' );
    pthS = concatenateTags( parentConstrArgs{ iOseEmbComponent } );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outSourceComponentName' ) 
            if ~isSet
                pthS{ 1 } = varargin{ i + 1 };
                break
            else
                error( 'outSourceComponentName has been already set' )
            end
        end  
    end
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'outSourceRealizationName' )
            if ~isSet
                pthS{ 2 } = varargin{ i + 1 };
                break
            else
                error( 'outSourceRealizationName has been already set' )
            end
        end  
    end
    pthS = strjoin_e( pthS, '_' );
    pth = strjoin_e( { pth, ... 
                     getTag( parentConstrArgs{ iOutPDistance } ) , ...
                     getTag( parentConstrArgs{ iOutSDistance } ), ...
                     getTag( parentConstrArgs{ iOutDiffOp } ), ...
                     pthS, }, '_' ); 
end
for iR = nR : -1 : 1
    for iC = nCT : -1 : 1
        oseErrComponent( iC, iR )  = nlsaEmbeddedComponent_xi_d( ...
            parentConstrArgs{ iOutTrgEmbComponent }( iC, iR ) );
        pthSet = fullfile( getPath( parentConstrArgs{ iOseEmbComponent }( iC, iR ) ), pth );
        if isa( propVal{ iOseRefComponent }, 'nlsaEmbeddedComponent_rec' )
            pthSet = fullfile( pthSet, ...
                               getBasisFunctionTag( propVal{ iOseRefComponent }( iC, iR ) ) );

        end 
        oseErrComponent( iC, iR ) = setPath( oseErrComponent( iC, iR ), ...
                    fullfile( getPath( ...
                    parentConstrArgs{ iOseEmbComponent }( iC, iR ) ), ...
                    pth ) );

        oseErrComponent( iC, iR ) = setDefaultTag( oseErrComponent( iC, iR ) );
        oseErrComponent( iC, iR ) = setDefaultSubpath( oseErrComponent( iC, iR ) );
        oseErrComponent( iC, iR ) = setDefaultFile( oseErrComponent( iC, iR ) );

    end
end
mkdir( oseErrComponent )


%% ISR STATE AND VELOCITY ERROR
% Determine error-specific directories
pth = concatenateTags( parentConstrArgs{ iTrgEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetComponentName' ) 
        if ~isSet
            pth{ 1 } = varargin{ i + 1 };
            break
        else
            error( 'targetComponentName has been already set' )
        end
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetRealizationName' )
        if ~isSet
            pth{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'targetRealizationName has been already set' )
        end
    end  
end
pth = strjoin_e( pth, '_' );
if isa( propVal{ iIsrRefComponent }, 'nlsaEmbeddedComponent_rec' );
    pthS = concatenateTags( parentConstrArgs{ iIsrEmbComponent } );
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'sourceComponentName' ) 
            if ~isSet
                pthS{ 1 } = varargin{ i + 1 };
                break
            else
                error( 'sourceComponentName has been already set' )
            end
        end  
    end
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'sourceRealizationName' )
            if ~isSet
                pthS{ 2 } = varargin{ i + 1 };
                break
            else
                error( 'sourceRealizationName has been already set' )
            end
        end  
    end
    pthS = strjoin_e( pthS, '_' );
    pth = strjoin_e( pth, ... 
                   { getTag( parentConstrArgs{ iPDistance } ) , ...
                     getTag( parentConstrArgs{ iSDistance } ), ...
                     getTag( parentConstrArgs{ iDiffOp } ), ...
                     pthS }, '_' ); 
end
for iR = nR : -1 : 1
    for iC = nCT : -1 : 1
        isrErrComponent( iC, iR )  = nlsaEmbeddedComponent_xi_d( ...
            parentConstrArgs{ iTrgEmbComponent }( iC, iR ) );
        if isa( propVal{ iOseRefComponent }, 'nlsaEmbeddedComponent_rec' )
            pthSet = fullfile( pthSet, ...
                               getBasisFunctionTag( propVal{ iIsrRefComponent }( iC, iR ) ) );

        end 
        isrErrComponent( iC, iR ) = setPath( isrErrComponent( iC, iR ), ...
                    fullfile( getPath( propVal{ iIsrEmbComponent }( iC, iR ) ), pth ) );

        isrErrComponent( iC, iR ) = setDefaultTag( isrErrComponent( iC, iR ) );
        isrErrComponent( iC, iR ) = setDefaultSubpath( isrErrComponent( iC, iR ) );
        isrErrComponent( iC, iR ) = setDefaultFile( isrErrComponent( iC, iR ) );

    end
end
mkdir( isrErrComponent )


% COLLECT CONSTRUCTOR ARGUMENTS
constrArgs                = cell( 1, 2 * nnz( ifProp ) );
constrArgs( 1 : 2 : end ) = propName( ifProp );
constrArgs( 2 : 2 : end ) = propVal( ifProp );
constrArgs = [ { 'isrErrComponent' isrErrComponent } ...
               { 'oseErrComponent' oseErrComponent } ...
                 constrArgs parentConstrArgs ];
