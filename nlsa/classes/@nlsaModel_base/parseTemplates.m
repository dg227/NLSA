function constrArgs = parseTemplates( varargin ) 
%PARSETEMPLATES Construct property name-value pairs for the nlsaModel_base 
%   class constructor from templates.
%
%   The input arguments of parseTemplates are passed as name-value pairs using
%   the syntax:
%
%   constrArgs = parseTemplates( propName1, propVal1, propName2, propVal2, ... )
%
%   The following input names can be specified:
%
%   'sourceComponent': An [ nC nR ]-sized array of nlsaComponent objects 
%      specifying the source data. nC is the number of components (physical
%      variables) in the dataset and nR the number of realizations (ensemble 
%      members). This is the only mandatory argument in the function call.
%      The remaining arguments are assigned to default values if not set by 
%      the caller. 
%
%   'sourceTime': An [ 1 nR ]-sized cell array of vectors specifying the time
%      stamps of each sample in the dataset. The number of elements in 
%      sourceTime{ iR } must be equal to the number of samples in 
%      srcComponent( :, iR ). If 'sourceTime' is not specified it is set to the
%      default values time{ iR } = 1 : nS( iR ), where nS( iR ) is the 
%      number of samples in the iR-th realization.
% 
%   'timeFormat': A string specifying a valid Matlab date format for
%      conversion between numeric and string time data.
%      
%   'embeddingTemplate': an array of nlsaEmbeddedComponent objects 
%      specifying templates according to which the data in srcComponent 
%      are to be delay-embedded. embeddingTemplate must be either a scalar or a 
%      vector of size [ nC 1 ]. In the former case, it is assumed that every 
%      component in the dataset should be embedded using the same template. 
%      In the latter case, the embedding template is replicated nR times to 
%      match the number of realizations If 'embeddingTemplate' is not assigned
%      by the caller, it is set to the default case of no delay embedding.
%
%   'embeddingOrigin': A scalar or vector of size [ 1 nR ] specifying the
%      starting time index in the raw data to perform time-lagged
%      embedding. The embedding origins must be at least as large as the
%      embedding window of each component. If 'embeddingOrigin' is not 
%      specified, the time origin for each realization is set to the 
%      minimum possible value consistent with the embedding templates.
%
%   'embeddingPartition': An [ 1 nR ]-sized vector, partition, of nlsaPartition
%      objects specifying how each realization of the delay-embedded data is to
%      be partitioned. That is, in the resulting nlsaModel_base object, the
%      properties embComponent( iC, iR ).partition and 
%      trgEmbComponent( iC, iR ).partition are both set to partition( iR ) 
%      for all iC and iCT. The number of samples in partition( iR ) corresponds
%      the number of samples in the iR-th realization after delay embedding. 
%
%   'embeddingPartitionT': As in 'embeddingPartition', but allows for a
%      different "test" partition, operating along the column dimension of the
%      pairwise distance matrix, to accelerate batch-wise pairwise distance
%      calculation.
%
%   'embeddingPartitionQ': As in 'denEmbeddingPartitionQ', but for a "query" 
%      partition, operating along the row dimension of the pairwise distance 
%      matrix. 
%
%   'targetComponent': An [ nCT nR ]-sized array of nlsaComponent objects 
%      specifying the target data. nCT is the number of target components. 
%      the number of realizations nR must be equal to the number of
%      realizations in the source data. If 'trgComponent' is not
%      specified, it is set equal to 'srcComponent'. 
% 
%   'targetEmbeddingOrigin', 'targetEmbeddingTemplate': Same as 
%   'embeddingTemplate' and 'embeddingOrigin', respectively, but for the target%    data.
%
%   'sourceRealizationName': A string which, if defined, is used to replace the
%      default directory name of the embeddingPartitionT data. This option is
%      useful to avoid long directory names in datasets with several
%      realizations, but may lead to non-uniqueness of the filename structure
%      and overwriting of results. 
%
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2019/11/23


%% CONSTRUCTOR PROPERTY LIST
propName = nlsaModel_base.listConstructorProperties; 
propVal  = cell( size( propName ) );
ifProp   = false( size( propName ) );
idxProp  = zeros( size( propName ) );
nProp    = numel( idxProp );

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
% Set time format
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'tFormat' )
        iTFormat = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'timeFormat' )
        if ~isempty( propVal{ iTFormat } )
            error( 'Time format has been already specified' )
        end
        propVal{ iTFormat } = varargin{ i + 1 };
        ifProp( iTFormat )  = true;
    end
end

% Set source data timestamps
% Timestamps will be set to the default values by the class constructor if no caller input 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'srcTime' )
        iSrcTime = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceTime' )                   
        if ~isempty( propVal{ iSrcTime } )
            error( 'Source time data have been already specified' )
        end
        propVal{ iSrcTime } = varargin{ i + 1 };
        ifProp( iSrcTime )  = true;
    end
end

% Set target time data
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'trgTime' )
        iTrgTime = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetTime' )                   
        if ~isempty( propVal{ iTrgTime } )
            error( 'Source time data have been already specified' )
        end
        propVal{ iTrgTime } = varargin{ i + 1 };
        ifProp( iTrgTime )  = true;
    end
end
if isempty( propVal{ iTrgTime } ) ...
   && ~isempty( propVal{ iSrcTime } )
    propVal{ iTrgTime } = propVal{ iSrcTime };
    ifProp( iTrgTime ) = true;
end

%% SOURCE DATA
% Import source data
% Compatibility of the source data will be tested by the class constructor
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'srcComponent' )
        iSrcComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'sourceComponent' )
        if ~isempty( propVal{ iSrcComponent } )
            error( 'Source components have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaComponent' )
            error( 'Source data must be specified as an array of nlsaComponent objects' )
        end
        propVal{ iSrcComponent } = varargin{ i + 1 };
        ifProp( iSrcComponent )  = true;
    end
end     
if isempty( propVal{ iSrcComponent } )
    error( 'Source components must be specified.' )
end
[ nC, nR ] = size( propVal{ iSrcComponent } );

%% SOURCE EMBEDDED DATA
% Parse embedding templates   
for iProp = 1 : nProp
    if strcmp( propName{ iProp }, 'embComponent' )
        iEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingTemplate' )
        if ~isempty( propVal{ iEmbComponent } ) 
            error( 'Time-lagged embedding templates have been already specified' )
        end
        if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
            && isscalar( varargin{ i + 1 } )
            propVal{ iEmbComponent } = repmat( varargin{ i + 1 }, [ nC 1 ] );
        elseif    isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nC
            propVal{ iEmbComponent } = varargin{ i + 1 };
            if size( propVal{ iEmbComponent }, 2 ) > 1 
                propVal{ iEmbComponent } = propVal{ iEmbComponent }';
            end
        else
            error( 'Time-lagged embedding templates must be specified as a scalar nlsaEmbeddedComponent object, or vector of nlsaEmbeddedComponent objects of size equal to the number of components' )
        end
        ifProp( iEmbComponent )  = true;
    end
end
if isempty( propVal{ iEmbComponent } )
    for iC = nC : -1 : 1
        propVal{ iEmbComponent }( iC ) = nlsaEmbeddedComponent_e();
    end
    propVal{ iEmbComponent } = propVal{ iEmbComponent }';
    ifProp( iEmbComponent ) = true;
end
for iC = 1 : nC
    propVal{ iEmbComponent }( iC ) = setDimension( propVal{ iEmbComponent }( iC ), ...
                                                   getDimension( propVal{ iSrcComponent }( iC ) ) );
end

% Determine minimum embedding origin from template
minEmbeddingOrigin = getMinOrigin( propVal{ iEmbComponent } );

% Replicate template to form embedded component array
propVal{ iEmbComponent } = repmat( propVal{ iEmbComponent }, [ 1 nR ] );             


% Parse embedding origin templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingOrigin' )
        if isSet
            error( 'Time-lagged embedding origins have been already specified' )
        end
        if ispsi( varargin{ i + 1 } )
            embeddingOrigin = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isvector( varargin{ i + 1 } ) && numel( varargin{ i + 1 } ) == nR 
            embeddingOrigin = varargin{ i + 1 };
        end
        isSet = true;
    end
end
if ~isSet
    embeddingOrigin = minEmbeddingOrigin * ones( 1, nR );
end
for iR = 1 : nR
    for iC = 1 : nC
        propVal{ iEmbComponent }( iC, iR ) = setOrigin( propVal{ iEmbComponent }( iC, iR ), embeddingOrigin( iR ) );
    end
end

% Determine maximum number of samples in each realization after embedding
maxNSRE = zeros( 1, nR );
for iR = 1 : nR
    maxNSRE( iR ) = getMaxNSample( propVal{ iEmbComponent }( :, iR ), ...
                                   getNSample( propVal{ iSrcComponent }( :, iR ) ) );
end

% Parse partition templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingPartition' )
        if isSet
            error( 'The partition for the embedded data has been already specified' )
        end
        if isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
           && isscalar( varargin{ i + 1 } )
            partition = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nR
            partition = varargin{ i + 1 };
        else
            error( 'Data partitions must be specified as a scalar nlsaPartition object, or a vector of nlsaPartition objects of size equal to the number of ensemble realizations' )
        end
        isSet = true;
    end
end 
if ~isSet
    for iR = nR : -1 : 1
        partition( iR ) = nlsaPartition( 'nSample', maxNSRE( iR ) );
    end
end
for iR = 1 : nR
    if getNSample( partition( iR ) ) > maxNSRE( iR )
        error( 'Number of time-lagged embedded samples is above maximum value' )
    end
    for iC = 1 : nC
        propVal{ iEmbComponent }( iC, iR ) = setPartition( propVal{ iEmbComponent }( iC, iR ), partition( iR ) );
    end 
end
nSRE   = getNSample( partition ); % Number of samples in each realization after embedding
nSE = sum( nSRE );


% Setup embedded component tags, directories, and filenames
for iR = 1 : nR
     
    for iC = 1 : nC
        propVal{ iEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iEmbComponent }( iC, iR ) );
        propVal{ iEmbComponent }( iC, iR ) = ...
            setComponentTag( ...
                propVal{ iEmbComponent }( iC, iR ), ...
                getComponentTag( propVal{ iSrcComponent }( iC, 1 ) ) );
        propVal{ iEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iSrcComponent }( 1, iR ) ) );

        tag  = getTag( propVal{ iEmbComponent }( iC, iR ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );
        
        propVal{ iEmbComponent }( iC, iR ) = ...        
            setPath( propVal{ iEmbComponent }( iC, iR ), pth );

        propVal{ iEmbComponent }( iC, iR ) = ...
            setDefaultSubpath( propVal{ iEmbComponent }( iC, iR ) );


        propVal{ iEmbComponent }( iC, iR ) = ...
            setDefaultFile( propVal{ iEmbComponent }( iC, iR ) );

    end
end
mkdir( propVal{ iEmbComponent } )
          
% If requested, create "test" embedded components

% Parse "test" partition templates.
% Test partition must be a coarsening of the partition in the embComponent 
% property.
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingPartitionT' )
        if isSet
            error( 'The test partition for the embedded data has been already specified' )
        end
        if ~(    isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
              && isscalar( varargin{ i + 1 } ) ...
              && isFiner( partition, varargin{ i + 1 } ) ) 
           error( 'Test embedded data partition must be specified as a scalar nlsaPartition object, which is a coarseing of the embedded data partition.' )
        end
        partitionT = varargin{ i + 1 };
        isSet = true;
    end
end 

% If test partition was provided, create an embComponentT property; otherwise
% it will be set it to empty by the class constructor.
if isSet
    for iProp = 1 : nProp
        if strcmp( propName{ iProp }, 'embComponentT' )
            iEmbComponentT = iProp;
            break
        end
    end

    if isa( propVal{ iEmbComponent }, 'nlsaEmbeddedComponent_xi' )
        propVal{ iEmbComponentT }( nC, 1 ) = nlsaEmbeddedComponent_xi_e();
    else
        propVal{ iEmbComponentT }( nC, 1 ) = nlsaEmbeddedComponent_e();
    end
    propVal{ iEmbComponentT } = mergeCol( propVal{ iEmbComponentT }, ...
                                          propVal{ iEmbComponent }, ...
                                          'partition', partitionT ); 
    ifProp( iEmbComponentT ) = true;
    % Check if we should compress realization tags
    isSet2 = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'sourceRealizationName' )
            if ~isSet2
                propVal{ iEmbComponentT } = setRealizationTag( ...
                    propVal{ iEmbComponentT }, varargin{ i + 1 } );
                isSet2 = true;
                break
            else
                error( 'sourceRealizationName has been already set' )
            end
        end
    end  
    for iC = 1 : nC 
        tag  = getTag( propVal{ iEmbComponentT }( iC ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );
        propVal{ iEmbComponentT }( iC ) = ...        
            setPath( propVal{ iEmbComponentT }( iC ), pth );
    end
    mkdir( propVal{ iEmbComponentT } )
end

% If requested, create "query" embedded components

% Parse "query" partition templates.
% Query partition must be a coarsening of the partition in embComponent. 
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'embeddingPartitionQ' )
        if isSet
            error( 'The query partition for the embedded data has been already specified' )
        end
        if ~(    isa( varargin{ i + 1 }, 'nlsaPartition' ) ...
              && isscalar( varargin{ i + 1 } ) ...
              && isFiner( partition, varargin{ i + 1 } ) ) 
           error( 'Query embedded data partition must be specified as a scalar nlsaPartition object, which is a coarseing of the embedded data partition.' )
        end
        partitionQ = varargin{ i + 1 };
        isSet = true;
    end
end 

% If query partition was provided create an embComponentQ property; otherwise 
% it will be set to empty by the class constructor. 
if isSet
    for iProp = 1 : nProp
        if strcmp( propName{ iProp }, 'embComponentQ' )
            iEmbComponentQ = iProp;
            break
        end
    end

    if isa( propVal{ iEmbComponent }, 'nlsaEmbeddedComponent_xi' )
        propVal{ iEmbComponentQ }( nC, 1 ) = nlsaEmbeddedComponent_xi_e();
    else
        propVal{ iEmbComponentQ }( nC, 1 ) = nlsaEmbeddedComponent_e();
    end
    propVal{ iEmbComponentQ } = mergeCol( propVal{ iEmbComponentQ }, ...
                                          propVal{ iEmbComponent }, ...
                                          'partition', partitionQ ); 
    ifProp( iEmbComponentQ ) = true;
    % Check if we should compress realization tags
    isSet2 = false;
    for i = 1 : 2 : nargin
        if strcmp( varargin{ i }, 'sourceRealizationName' )
            if ~isSet2
                propVal{ iEmbComponentQ } = setRealizationTag( ...
                    propVal{ iEmbComponentQ }, varargin{ i + 1 } );
                isSet2 = true;
                break
            else
                error( 'sourceRealizationName has been already set' )
            end
        end
    end  
    for iC = 1 : nC 
        tag  = getTag( propVal{ iEmbComponentQ }( iC ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );
        propVal{ iEmbComponentQ }( iC ) = ...        
            setPath( propVal{ iEmbComponentQ }( iC ), pth );
    end
    mkdir( propVal{ iEmbComponentQ } )
end


%% TARGET DATA

% Import target data and determine the number of samples and dimension 
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'trgComponent' )
        iTrgComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetComponent' )
        if ~isempty( propVal{ iTrgComponent } )
            error( 'Target components have been already specified' )
        end
        if ~isa( varargin{ i + 1 }, 'nlsaComponent' )
            error( 'Target data must be specified as an array of nlsaComponent objects' )
        end
        propVal{ iTrgComponent } = varargin{ i + 1 };
        ifProp( iTrgComponent )  = true;
    end
end
if isempty( propVal{ iTrgComponent } )
    propVal{ iTrgComponent } = propVal{ iSrcComponent };
    ifProp( iTrgComponent ) = true;
end     
nCT = size( propVal{ iTrgComponent }, 1 );
if size( propVal{ iTrgComponent }, 2 ) ~= nR
    error( 'The number of source and target realizations must be equal' )
end

%% TARGET EMBEDDED DATA
% Parse embedding templates for the target data   
for iProp = 1 : nProp 
    if strcmp( propName{ iProp }, 'trgEmbComponent' )
        iTrgEmbComponent = iProp;
        break
    end
end
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetEmbeddingTemplate' )
        if ~isempty( propVal{ iTrgEmbComponent } ) 
            error( 'Time-lagged embedding templates for the target data have been already specified' )
        end
        if  isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
            && isscalar( varargin{ i + 1 } )
            propVal{ iTrgEmbComponent } = repmat( ...
                varargin{ i + 1 }, [ nCT 1 ] );
        elseif isa( varargin{ i + 1 }, 'nlsaEmbeddedComponent' ) ...
               && isvector( varargin{ i + 1 } ) ...
               && numel( varargin{ i + 1 } ) == nCT
            propVal{ iTrgEmbComponent } = varargin{ i + 1 };
            if size( propVal{ iTrgEmbComponent }, 2 ) > 1 
                propVal{ iTrgEmbComponent } = propVal{ iTrgEmbComponent }';
            end
        else
            error( 'Time-lagged embedding templates must be specified as a scalar nlsaEmbeddedComponent object, or vector of nlsaEmbeddedComponent objects of size equal to the number of components' )
        end
        ifProp( iTrgEmbComponent ) = true;
    end
end
if isempty( propVal{ iTrgEmbComponent } )
    propVal{ iTrgEmbComponent } = propVal{ iEmbComponent };
    ifProp( iTrgEmbComponent )  = true;
end

for iC = 1 : nCT
    propVal{ iTrgEmbComponent }( iC ) = setDimension( propVal{ iTrgEmbComponent }( iC ), ...
                                        getDimension( propVal{ iTrgComponent }( iC ) ) );
end

% Determine time limits for embedding origin 
minEmbeddingOrigin = getMinOrigin( propVal{ iTrgEmbComponent } );

% Replicate template to form target embedded component array
propVal{ iTrgEmbComponent } = repmat( propVal{ iTrgEmbComponent }, [ 1 nR ] );             
% Parse embedding origin templates
isSet = false;
for i = 1 : 2 : nargin
    if strcmp( varargin{ i }, 'targetEmbeddingOrigin' )
        if isSet
            error( 'Time-lagged embedding origins for the target data have been already specified' )
        end
        if ispsi( varargin{ i + 1 } )
            embeddingOrigin = repmat( varargin{ i + 1 }, [ 1 nR ] );
        elseif isvector( varargin{ i + 1 } ) && numel( varargin{ i + 1 } ) == nR 
            embeddingOrigin = varargin{ i + 1 };
        end
        isSet = true;
    end
end
if ~isSet
    embeddingOrigin = max( minEmbeddingOrigin, ...
                           getOrigin( propVal{ iEmbComponent }( 1, 1 ) ) ) ...
                    * ones( 1, nR );
end
for iR = 1 : nR
    if embeddingOrigin( iR ) < minEmbeddingOrigin
        error( 'Time-lagged embedding origin is below minimum value' )
    end
    for iC = 1 : nCT
        propVal{ iTrgEmbComponent }( iC, iR ) = setOrigin( propVal{ iTrgEmbComponent }( iC, iR ), embeddingOrigin( iR ) );
    end
end

% Determine maximum number of samples in each embedded component
maxNSRET = zeros( 1, nR );
for iR = 1 : nR
    maxNSRET( iR ) = getMaxNSample( propVal{ iTrgEmbComponent }( :, iR ), ...
                                    getNSample( propVal{ iTrgComponent }( :, iR ) ) );
end

% Check number of samples in target embedded data
for iR = 1 : nR
    if getNSample( partition( iR ) ) > maxNSRET( iR )
         msgStr = [ 'Number of time-lagged embedded samples ', ...
                    int2str( getNSample( partition( iR ) ) ), ...
                    ' is above maximum value ', ...
                    int2str( maxNSRET( iR ) ) ];
        error( msgStr ) 
    end
    for iC = 1 : nCT
        propVal{ iTrgEmbComponent }( iC, iR ) = setPartition( propVal{ iTrgEmbComponent }( iC, iR ), partition( iR ) );
    end 
end

% Setup target embedded component tags, directories, and filenames
for iR = 1 : nR
    for iC = 1 : nCT

        propVal{ iTrgEmbComponent }( iC, iR ) = ...
            setDefaultTag( propVal{ iTrgEmbComponent }( iC, iR ) );
        propVal{ iTrgEmbComponent }( iC, iR ) = ...
            setComponentTag( ...
                propVal{ iTrgEmbComponent }( iC, iR ), ...
                getComponentTag( propVal{ iTrgComponent }( iC, 1 ) ) );
        propVal{ iTrgEmbComponent }( iC, iR ) = ...
            setRealizationTag( ...
                propVal{ iTrgEmbComponent }( iC, iR ), ...
                getRealizationTag( propVal{ iTrgComponent }( 1, iR ) ) );

        tag  = getTag( propVal{ iTrgEmbComponent }( iC, iR ) );
        pth  = fullfile( modelPath, 'embedded_data', ...
                         strjoin_e( tag, '_' ) );
        
        propVal{ iTrgEmbComponent }( iC, iR ) = ...
            setPath( propVal{ iTrgEmbComponent }( iC, iR ), pth );

        propVal{ iTrgEmbComponent }( iC, iR ) = ...
            setDefaultSubpath( propVal{ iTrgEmbComponent }( iC, iR ) );


        propVal{ iTrgEmbComponent }( iC, iR ) = ...
            setDefaultFile( propVal{ iTrgEmbComponent }( iC, iR ) );
    end
end
mkdir( propVal{ iTrgEmbComponent } )


%% COLLECT ARGUMENTS
constrArgs                = cell( 1, 2 * nnz( ifProp ) );
constrArgs( 1 : 2 : end ) = propName( ifProp );
constrArgs( 2 : 2 : end ) = propVal( ifProp );
