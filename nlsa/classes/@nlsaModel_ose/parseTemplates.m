function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel_ose class
%   constructor from templates.
%
%   The arguments of parseTemplates are passed as property name-property value
%   pairs using the syntax:
%
%   constrArgs  = parseTemplates( propName1, propVal1, propName2, propVal2, ... ).
%
%   The following properties can be specified in addition to those available in
%   the parseTemplates method of the nlsaModel superclass:
%
%   'outComponent': An [ nC nRO ]-sized array of nlsaComponent objects 
%      specifying the out-of-sample source data. nC is the number of 
%      components (physical variables) and nRO the number of realizations
%      (ensemble members) in the out-of-sample data. nC must be equal to the
%      number of components in the in-sample source data as specified in
%      'srcComponent'.
%
%   'outTime': An [ 1 nRO ]-sized cell array of vectors specifying the time
%      stamps of each sample in the OSE dataset. The number of elements in 
%      outTime{ iR } must be equal to the number of samples in 
%      outComponent( :, iR ). If 'time' is not specified it is set to the
%      default values time{ iR } = 1 : nSO( iR ), where nSO( iR ) is the 
%      number of samples in the iR-th realization.
% 
%   'outTimeFormat': A string specifying a valid Matlab date format for
%      conversion between numeric and string time data.
%      
%   'outEmbeddingTemplate': An array of nlsaEmbeddedComponent objects 
%      specifying templates according to which the data in outComponent 
%      are to be time-lagged embedded. embComponent must be either a 
%      scalar or a vector of size [ nC 1 ]. In the former case, it is 
%      assumed that every component in the dataset should be embedded using 
%      the same template. If 'outEmbeddingTemplate' is not assigned by the 
%      caller, it is set to the default case of no time-lagged embedding.
%
%   'outEmbeddingOrigin': A scalar or vector of size [ 1 nRO ] specifying the
%      starting time index in the raw data to perform time-lagged
%      embedding. The embedding origins must be at least as large as the
%      embedding window of each component. If 'outEmbeddingOrigin' is not
%      specified, the time origin for each realization is set to the minimum
%      possible value consistent with the embedding templates.
%
%   'outEmbeddingPartition': An [ 1 nRO ]-sized vector of nlsaPartition objects
%      specifying how each realization in the OSE dataset is to be partitioned.
%      The number of samples in partition( iR ) must be equal to nSOE( iR ),
%      where nSOE( iR ) is the number of samples in the iR-th realization
%      of the out-of-sample data after time lagged embedding. 
%
%   'osePairwiseDistanceTemplate': An nlsaPairwiseDistance object specifying
%      the method and parameters (e.g., number of nearest neighbors) used 
%      to compute the graph edge weights for the dataset. If 
%      'osePairwiseDistanceTemplate' is not specified, it is set to the same
%      pairwise distance as that used for the in-sample data.
% 
%   'oseDiffusionOperatorTemplate': An nlsaDiffusionOperator_ose object
%       specifying the OSE operator in the data analysis model.
%
%   'oseEmbeddingTemplate': An array of nlsaEmbeddedComponent objects 
%      specifying templates for the out-of-sample extension of the target 
%      data. 'oseEmbeddingTemplate' must be either a scalar or a vector of size
%      [ nCT 1 ]. In the former case, it is assumed that every component in
%      the dataset should be out-of-sample extended using the same template.
%      If 'oseEmbeddingTemplate' is not assigned by the caller, it is set to
%      the default case of out-of-sample extension via kernel averaging 
%      operators (as opposed to Nystrom).
%
%   'oseReconstructionPartition': An [ 1 nRO ]-sized vector of nlsaPartition
%      objects specifying how each realization of the OSE reconstructed data is
%      to be partitioned. That is, in the resulting nlsaModel_base object, the
%      property oseRecComponent( iC, iR ).partition is  set to partition( iR ) 
%      for all iC and iR. The number of samples in partition( iR ) must not
%      exceed the number of samples in the iR-th realization of the out of 
%      sample delay-embedded data, plus nE( iCT ) - 1, where nE( iCT ) is the
%      number of delays for target component iCT. If 'reconstructionPartition'
%      is not specified, the partition is set to a single-batch partition with
%      the maximum number of samples allowed. 
% 
%   'outComponentName': A string which, if defined, is used to replace the
%      default directory name of the OSE pairwise distance data. This option is
%      useful to avoid long directory names in datasets with several
%      components, but may lead to non-uniqueness of the filename structure
%      and overwriting of results. 
%
%   'outRealizationName': Similar to 'outComponentName', but used to compress
%      the realization-dependent part of the pairwise distance directory.
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2020/08/01    


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel_ose.listConstructorProperties;         
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );

%% SUPERCLASS CONSTRUCTOR ARGUMENTS
parentConstrArgs = nlsaModel.parseTemplates( varargin{ : } );
for iProp = 1 : 2 : numel( parentConstrArgs )
    switch parentConstrArgs{ iProp }
        case 'srcComponent' % Needed in case no OS data are specified 
            iSrcComponent = iProp + 1;
        case 'embComponent'
            iEmbComponent = iProp + 1;
        case 'pairwiseDistance' % Needed for the partitions of the test (in-sample) data
            iPDist = iProp + 1;
        case 'diffusionOperator'  % Needed to get the directory of the diffusion operator
            iDiffOp = iProp + 1;
        case 'trgEmbComponent' 
            iTrgEmbComponent = iProp + 1;
        case 'trgComponent'
            iTrgComponent = iProp + 1;
    end
end 
nCT = size( parentConstrArgs{ iTrgEmbComponent }, 1 );


% Needed in case the corresponding templates for the OS data are not specified
for i = 1 : 2 : nargin
    switch varargin{ i }
        case 'embeddingTemplate'
            iEmbTemplate = i + 1;
    end    
end

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


%% TIME SPECIFICATION
% Set OS time format
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'outTimeFormat' )
        iOutTimeFormat = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTimeFormat' )
        if ~isempty( propVal{ iOutTimeFormat } )
            error( 'Time format for OSE data has been already specified' )
        end
        propVal{ iOutTimeFormat } = varargin{ i + 1 };
        ifProp( iOutTimeFormat )  = true;
    end
end

% Set OS data timestamps
% Timestamps will be set to ithe default values by the class constructor if no caller input
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outTime' )
        iOutTime = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTime' )                   
        if ~isempty( propVal{ iOutTime } )
            error( 'OSE time data have been already specified' )
        end
        propVal{ iOutTime } = varargin{ i + 1 };
        ifProp( iOutTime )  = true;
    end
end


%% OS DATA
% Import OS data
% Compatibility of the OS data with the source data will be determined by the class constructor 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outComponent' )
        iOutComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outComponent' )
        if ~isempty( propVal{ iOutComponent } )
            error( 'ose components have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaComponent' )
            error( 'OS data must be specified as an array of nlsaComponent objects' )
        end
        propVal{ iOutComponent } = varargin{ i + 1 };
        ifProp( iOutComponent )  = true;
    end
end
if isempty( propVal{ iOutComponent } )
    propVal{ iOutComponent } = parentConstrArgs{ iSrcComponent };
end     
[ nC, nRO ] = size( propVal{ iOutComponent } );


%% OS EMBEDDED DATA
% Parse embedding templates for the ose data   
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outEmbComponent' )
        iOutEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outEmbeddingTemplate' )
        if ~isempty( propVal{ iOutEmbComponent } ) 
            error( 'Time-lagged embedding templates for the ose data have been already specified' )
        end
        if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
            && isscalar( varargin{ i + 1 } )
            propVal{ iOutEmbComponent } = repmat( varargin{ i + 1 }, [ nC 1 ] );
        elseif    isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nC
            propVal{ iOutEmbComponent } = varargin{ i + 1 };
            if isrow( propVal{ iOutEmbComponent } )  
                propVal{ iOutEmbComponent } = propval{ iOutEmbComponent }';
            end
        else
            error( 'Time-lagged embedding templates must be specified as a scalar nlsaEmbeddedComponent object, or vector of nlsaEmbeddedComponent objects of size equal to the number of components' )
        end
        ifProp( iOutEmbComponent ) = true;
    end
end
if isempty( propVal{ iOutEmbComponent } )
    propVal{ iOutEmbComponent } = parentConstrArgs{ iEmbComponent }( :, 1 );;
    ifProp( iOutEmbComponent )  = true;
else
    for iC = 1 : nC
        propVal{ iOutEmbComponent }( iC ) = setDimension( propVal{ iOutEmbComponent }( iC ), ...
                                                          getDimension( propVal{ iOutComponent }( iC ) ) );
    end
end

% Determine minimum embedding origin from template
minEmbeddingOrigin = getMinOrigin( propVal{ iOutEmbComponent } );

% Replicate template to form embeedded component array
propVal{ iOutEmbComponent } = repmat( propVal{ iOutEmbComponent }, [ 1 nRO ] );


% Parse os embedding origin templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outEmbeddingOrigin' )
        if isSet
            error( 'Time-lagged embedding origins for the OS data have been already specified' )
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
    for iC = 1 : nC
        propVal{ iOutEmbComponent }( iC, iR ) = setOrigin( propVal{ iOutEmbComponent }( iC, iR ), embeddingOrigin( iR ) );
    end
end

% Determine maximum number of samples in each embedded component
maxNSORE = zeros( 1, nRO );
for iR = 1 : nRO
    maxNSORE( iR ) = getMaxNSample( propVal{ iOutEmbComponent }( :, iR ), ...
                                    getNSample( propVal{ iOutComponent }( :, iR ) ) );
end

% Parse partition templates for the OS embedded data
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outEmbeddingPartition' )
        if isSet
            error( 'Partition templates for the OS data have already been specified' )
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
if ~isSet
    for iR = nRO : -1 : 1
        partition( iR ) = nlsaPartition( 'nSample', maxNSORE( iR ) );
    end
end

for iR = 1 : nRO
    if getNSample( partition( iR ) ) > maxNSORE( iR )
         msgStr = [ 'Number of time-lagged embedded samples ', ...
                    int2str( getNSample( partition( iR ) ) ), ...
                    ' is above maximum value ', ...
                    int2str( maxNSORE( iR ) ) ];
        error( msgStr ) 
    end
    for iC = 1 : nC
        propVal{ iOutEmbComponent }( iC, iR ) = setPartition( propVal{ iOutEmbComponent }( iC, iR ), partition( iR ) );
    end 
end

% Setup embedded component tags, directories, and filenames
for iR = 1 : nRO 
    for iC = 1 : nC
        propVal{ iOutEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iOutEmbComponent }( iC, iR ) );
        propVal{ iOutEmbComponent }( iC, iR ) = ...
            setComponentTag( ...
                propVal{ iOutEmbComponent }( iC, iR ), ...
                getComponentTag( propVal{ iOutComponent }( iC, 1 ) ) );
        propVal{ iOutEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iOutEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iOutComponent }( 1, iR ) ) );

        tag  = getTag( propVal{ iOutEmbComponent }( iC, iR ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );

        propVal{ iOutEmbComponent }( iC, iR ) = ...
            setPath( propVal{ iOutEmbComponent }( iC, iR ), pth );

        propVal{ iOutEmbComponent }( iC, iR ) = ...
            setDefaultSubpath( propVal{ iOutEmbComponent }( iC, iR ) );

        propVal{ iOutEmbComponent }( iC, iR ) = ...
            setDefaultFile( propVal{ iOutEmbComponent }( iC, iR ) );

    end
end
mkdir( propVal{ iOutEmbComponent } )

%% OS TARGET DATA

% Import out-of-sample target data and determine the number of samples 
% and dimension 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'outTrgComponent' )
        iOutTrgComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'outTargetComponent' )
        if ~isempty( propVal{ iOutTrgComponent } )
            error( 'Out-of-sample target components have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaComponent' )
            error( 'Out-of-sample target data must be specified as an array of nlsaComponent objects' )
        end
        if size( varargin{ i + 1 }, 2 ) ~= nRO
            error( 'The number of source and target OS realizations must be equal' )
        end
        propVal{ iOutTrgComponent } = varargin{ i + 1 };
        ifProp( iOutTrgComponent )  = true;
    end
end


%% OSE TARGET EMBEDDED DATA
if ifProp( iOutTrgComponent )
    % Parse embedding templates for the target data   
    for iProp = 1 : nProp 
        if strcmp( propName{ iProp }, 'outTrgEmbComponent' )
            iOutTrgEmbComponent = iProp;
            break
        end
    end
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'targetEmbeddingTemplate' )
            if ~isempty( propVal{ iOutTrgEmbComponent } ) 
                error( 'Time-lagged embedding templates for the target data have been already specified' )
            end
            if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
                && isscalar( varargin{ i + 1 } )
                propVal{ iOutTrgEmbComponent } = repmat( ...
                    varargin{ i + 1 }, [ nCT 1 ] );
            elseif isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
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
        propVal{ iOutTrgEmbComponent } = propVal{ iEmbComponent }( :, 1 );
        ifProp( iOutTrgEmbComponent )  = true;
    end

    for iC = 1 : nCT
        propVal{ iOutTrgEmbComponent }( iC ) = setDimension( ...
            propVal{ iOutTrgEmbComponent }( iC ), ...
            getDimension( propVal{ iOutTrgComponent }( iC ) ) );
    end

    % Determine time limits for embedding origin 
    minEmbeddingOrigin = getMinOrigin( propVal{ iOutTrgEmbComponent } );

    % Replicate template to form target embedded component array
    propVal{ iOutTrgEmbComponent } = repmat( propVal{ iOutTrgEmbComponent }, [ 1 nRO ] );             
    % Parse embedding origin templates
    isSet = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'targetEmbeddingOrigin' )
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
        embeddingOrigin = max( minEmbeddingOrigin, ...
                               getOrigin( propVal{ iOutEmbComponent }( 1, 1 ) ) ) ...
                        * ones( 1, nRO );
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
        
        maxNSRET( iR ) = getMaxNSample( ...
            propVal{ iOutTrgEmbComponent }( :, iR ), ...
            getNSample( propVal{ iOutTrgComponent }( :, iR ) ) );
    end

    % Check number of samples in target embedded data
    for iR = 1 : nRO
        if getNSample( partition( iR ) ) > maxNSRET( iR )
             msgStr = [ 'Number of time-lagged embedded samples ', ...
                        int2str( getNSample( partition( iR ) ) ), ...
                        ' is above maximum value ', ...
                        int2str( maxNSRET( iR ) ) ];
            error( msgStr ) 
        end
        for iC = 1 : nCT
            propVal{ iOutTrgEmbComponent }( iC, iR ) = setPartition( propVal{ iOutTrgEmbComponent }( iC, iR ), partition( iR ) );
        end 
    end

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
end

%% OSE PAIRWISE DISTANCE
% Parse OSE distance template and set OSE distance partition
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'osePairwiseDistance' )
        iOsePDist = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'osePairwiseDistanceTemplate' )
        if ~isempty( propVal{ iOsePDist } )
            error( 'A pairwise distance template has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPairwiseDistance' ) ...
           && isscalar( varargin{ i + 1 } )
            propVal{ iOsePDist } = varargin{ i + 1 };
            ifProp( iOsePDist )  = true;
            ifRetainTag          = true;
        else
            error( 'The OSE pairwise distance template must be specified as a scalar nlsaPairwiseDistance object' )
        end
    end
end
if isempty( propVal{ iOsePDist } )
    propVal{ iOsePDist } = parentConstrArgs{ iPDist };
    ifProp( iOsePDist )  = true;
    ifRetainTag          = false;
end

% Partition for query (OSE) data
propVal{ iOsePDist } = setPartition( propVal{ iOsePDist }, partition );

% partition for test (in-sample) data 
propVal{ iOsePDist } = setPartitionTest( propVal{ iOsePDist }, ...
                                         getPartition( parentConstrArgs{ iPDist } ) ); 

if ifRetainTag
    tag = getTag( propVal{ iOsePDist } );
    if ~isempty( tag )
        tag = [ tag '_' ];
    end
else
    tag = [];
end
propVal{ iOsePDist } = setTag( propVal{ iOsePDist }, ...
        [ tag getDefaultTag( propVal{ iOsePDist } ) ] ); 

% Determine distance-specific directories
pth = concatenateTags( propVal{ iOutEmbComponent } );
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
            break
        end
    end  
end
pth = strjoin_e( pth, '_' );


% Determine path prefix for OSE pairwise distance from parent class

pthParent = concatenateTags( parentConstrArgs{ iEmbComponent } );
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceComponentName' ) 
        if ~isSet
            pthParent{ 1 } = varargin{ i + 1 };
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
            pthParent{ 2 } = varargin{ i + 1 };
            break
        else
            error( 'sourceRealizationName has been already set' )
        end
    end  
end
pthParent = strjoin_e( pthParent, '_' );


% Assign pairwise distance paths and filenames
modelPathDO = fullfile( modelPath, 'processed_data', ...
                        pthParent, pth, ...
                        getTag( propVal{ iOsePDist } ) ); 
propVal{ iOsePDist } = setPath( propVal{ iOsePDist }, modelPathDO ); 
propVal{ iOsePDist } = setDefaultSubpath( propVal{ iOsePDist } );
propVal{ iOsePDist } = setDefaultFile( propVal{ iOsePDist } );
mkdir( propVal{ iOsePDist } )


%% PARSE OSE DIFFUSION OPERATOR TEMPLATE 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'oseDiffusionOperator' )
        iOseDiffOp = iProp;
        break
    end  
end
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseDiffusionOperatorTemplate' )
        if ~isSet
            if isa( varargin{ i + 1 }, 'nlsaDiffusionOperator_ose' ) ...
               && isscalar( varargin{ i + 1 } )
                propVal{ iOseDiffOp } = varargin{ i + 1 };
                isSet = true;
            else
                error( 'The OSE diffusion operator template must be specified as a scalar nlsaDiffusionOperator_ose object' )
            end
        else       
            error( 'An OSE diffusion operator template has been already specified' )
        end
    end
end
if ~isSet
    propVal{ iOseDiffOp } = nlsaDiffusionOperator_ose( ...
        parentConstrArgs{ iDiffOp } );
end
ifProp( iOseDiffOp )  = true;
propVal{ iOseDiffOp } = setPartition( propVal{ iOseDiffOp }, partition );
propVal{ iOseDiffOp } = setPartitionTest( propVal{ iOseDiffOp }, ...
                                    getPartitionTest( propVal{ iOsePDist } ) );
tag = getTag( propVal{ iOseDiffOp } );
if ~isempty( tag )
    tag = [ tag '_' ];
end
propVal{ iOseDiffOp } = setTag( propVal{ iOseDiffOp }, ...
        [ tag getDefaultTag( propVal{ iOseDiffOp } ) ] ); 

% Assign diffusion operator paths and filenames
modelPathLO           = fullfile( getPath( parentConstrArgs{ iDiffOp } ), ...
                                  pth, ...
                                  getTag( propVal{ iOsePDist } ), ...
                                  getTag( propVal{ iOseDiffOp } ) );
propVal{ iOseDiffOp } = setDefaultSubpath( propVal{ iOseDiffOp } );
propVal{ iOseDiffOp } = setPath( propVal{ iOseDiffOp }, modelPathLO );
mkdir( propVal{ iOseDiffOp } )
propVal{ iOseDiffOp } = setDefaultFile( propVal{ iOseDiffOp } );

%% STATE AND VELOCITY OSE COMPONENTS
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'oseEmbComponent' )
        iOseEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseEmbeddingTemplate' )
        if ~isempty( propVal{ iOseEmbComponent } )
            error( 'OSE templates have been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent_ose_n' ) ...
               && isscalar( varargin{ i + 1 } )
            propVal{ iOseEmbComponent } = repmat( ...
                varargin{ i + 1 }, [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nCT
            propVal{ iOseEmbComponent } = varargin{ i + 1 };
            if size( propVal{ iOseEmbComponent }, 2 ) > 1 
                propVal{ iOseEmbComponent } = propVal{ iOseEmbComponent }';
            end
        else
            error( 'The OSE templates must be specified as a scalar nlsaEmbeddedComponent_ose_n object or a vector of nlsaEmbeddedComponent_ose_n objects with number of elements equal to the number of test components' )
        end
        if any( getMaxEigenfunctionIndex( varargin{ i + 1 } ) ...
                > getNEigenfunction( propVal{ iOseDiffOp } ) )
            error( 'Insufficient number of OSE diffusion eigenfunctions' )
        end
        for iC = nCT : -1 : 1 
            iCSet = min( iC, numel( varargin{ i + 1 } ) );
            propVal{ iOseEmbComponent }( iC, 1 ) = ...
                nlsaEmbeddedComponent_ose_n( ...
                    parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) );
            propVal{ iOseEmbComponent }( iC, 1 ) = ...
                setEigenfunctionIndices( ...
                    propVal{ iOseEmbComponent }( iC, 1 ), ...
                    getEigenfunctionIndices( ...
                        varargin{ i + 1 }( iCSet ) ) );
        end
        ifProp( iOseEmbComponent ) = true;
    end
end
if isempty( propVal{ iOseEmbComponent } )
    for iC = nCT : -1 : 1
        propVal{ iOseEmbComponent }( iC, 1 ) = ...
            nlsaEmbeddedComponent_ose( ...
                parentConstrArgs{ iTrgEmbComponent }( iC, 1 ) );
    end
    ifProp( iOseEmbComponent ) = true;
end

for iR = 1 : nRO
    for iC = 1 : nCT
        propVal{ iOseEmbComponent }( iC, iR ) = ...
            setPartition( ...
                propVal{ iOseEmbComponent }( iC, iR ), ...
                getPartition( propVal{ iOutEmbComponent }( 1, iR ) ) );
        propVal{ iOseEmbComponent }( iC, iR ) = setComponentTag( ...
            propVal{ iOseEmbComponent }( iC, iR ), ...
            getComponentTag( parentConstrArgs{ iTrgComponent }( iC, 1 ) ) );
        propVal{ iOseEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iOseEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iOutEmbComponent }( 1, iR ) ) );
        propVal{ iOseEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iOseEmbComponent }( iC, iR ) );
        tg = getTag( propVal{ iOseEmbComponent }( iC, iR ) );
        propVal{ iOseEmbComponent }( iC, iR ) = ...
            setPath( propVal{ iOseEmbComponent }( iC, iR ), ...
                fullfile( ...
                    modelPathLO, ...
                    strjoin_e( tg( 1 : 3 ), '_' ), ...
                    getOseTag( propVal{ iOseEmbComponent } ) ) );
        propVal{ iOseEmbComponent }( iC, iR ) = setDefaultSubpath( propVal{ iOseEmbComponent }( iC, iR ) );
        propVal{ iOseEmbComponent }( iC, iR ) = setDefaultFile( propVal{ iOseEmbComponent }( iC, iR ) );
    end
end
mkdir( propVal{ iOseEmbComponent } )


%% OSE RECONSTRUCTED COMPONENTS
% Parse reconstruction templates
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'oseRecComponent' )
        iOseRecComponent = iProp;
        break
    end
end
ifProp( iOseRecComponent ) = true;

for iR = nRO : -1 : 1
    for iC = nCT : -1 : 1
        propVal{ iOseRecComponent }( iC, iR ) = ...
            nlsaComponent_rec();
    end
end

% Determine maximum number of samples in each realization of the reconstructed
% data
maxNSRRO = zeros( 1, nRO );
for iR = 1 : nRO
    maxNSRRO( iR ) = getNSample( ...
                       propVal{ iOseEmbComponent }( 1, iR )  )...
                   + getEmbeddingWindow( ...
                       propVal{ iOseEmbComponent }( 1, iR ) ) ...
                   - 1;
end


% Parse reconstruction partition templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'oseReconstructionPartition' )
        if isSet
            error( 'Partition templates for the OSE reconstructed data have been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
           && isscalar( varargin{ i + 1 } )
            oseRecPartition = repmat( varargin{ i + 1 }, [ 1 nRO ] );
        elseif isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nRO
            oseRecPartition = varargin{ i + 1 };
        else
            error( 'Data partitions must be specified as a scalar nlsaPartition object, or a vector of nlsaPartition objects of size equal to the number of ensemble realizations' )
        end
        isSet = true;
    end
end
if ~isSet
    for iR = nRO : -1 : 1
        oseRecPartition( iR ) = nlsaPartition( 'nSample', maxNSRRO( iR ) );
    end
end
for iR = 1 : nRO
    if getNSample( oseRecPartition( iR ) ) > maxNSRRO( iR )
        error( 'Number of reconstructed samples is above maximum value' )
    end
    for iC = 1 : nCT
        propVal{ iOseRecComponent }( iC, iR ) = setPartition( propVal{ iOseRecComponent }( iC, iR ), oseRecPartition( iR ) );
    end
end

% Setup OSE reconstructed component tags, directories, and filenames
for iR = 1 : nRO
    for iC = 1 : nCT
        propVal{ iOseRecComponent }( iC, iR ) = setDimension( ...
           propVal{ iOseRecComponent }( iC, iR ), ...
           getDimension( parentConstrArgs{ iTrgComponent }( iC, iR ) ) );
        propVal{ iOseRecComponent }( iC, iR ) = setComponentTag( ...
            propVal{ iOseRecComponent }( iC, iR ), ...
            getComponentTag( propVal{ iOseEmbComponent }( iC, iR ) ) );
        propVal{ iOseRecComponent }( iC, iR ) = setRealizationTag( ...
            propVal{ iOseRecComponent }( iC, iR ), ...
            getRealizationTag( propVal{ iOseEmbComponent }( iC, iR ) ) ); 
        propVal{ iOseRecComponent }( iC, iR ) = setDefaultSubpath( propVal{ iOseRecComponent }( iC, iR ) );
        propVal{ iOseRecComponent }( iC, iR ) = setDefaultFile( propVal{ iOseRecComponent }( iC, iR ) );
        propVal{ iOseRecComponent }( iC, iR ) = setPath( ...
            propVal{ iOseRecComponent }( iC, iR ), ...
            getPath( propVal{ iOseEmbComponent }( iC, iR ) ) );
    end
end
mkdir( propVal{ iOseRecComponent } )


% COLLECT CONSTRUCTOR ARGUMENTS
constrArgs                = cell( 1, 2 * nnz( ifProp ) );
constrArgs( 1 : 2 : end ) = propName( ifProp );
constrArgs( 2 : 2 : end ) = propVal( ifProp );
constrArgs = [ constrArgs  parentConstrArgs ];
