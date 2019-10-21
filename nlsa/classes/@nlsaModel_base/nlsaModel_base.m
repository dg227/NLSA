classdef nlsaModel_base
%NLSAMODEL_BASE   Class definition and constructor of basic data components
%   in nlsaModel* objects
%
%   nlsaModel_base objects consist of "source" and "target" data. The source 
%   data are the data are generally used for unsupervised learning tasks such as
%   evaluation of pairwise distances, kernels, covariances, etc. The target 
%   data are the output data for supervised learning such as ordinary 
%   regression, kernel regression, etc. 
%
%   The source and target data are partitioned into "components" and
%   "realizations". Each components represents a different physical variable 
%   (e.g., temperature, pressure, velocity, ...), and each realization 
%   corresponds to an independent realization of the dynamical system generating
%   the data (e.g., realizations obtained from an ensemble of initial 
%   conditions).
%
%   In addition to the source and target data, nlsaModel_base objects provide
%   storage for Takens delay embedded versions of this data. In NLSA, most 
%   supervised and unsupervised learning tasks are performed on the delay 
%   embedded data rather than than the raw source and target data.    
% 
%   The class constructor arguments are passed as property name-property 
%   value pairs using the syntax
%
%   model = nlsaModel_base( propName1, propVal1, propName2, propVal2, ... ).
%
%   The following properties can be specified
%
%   'srcComponent': An [ nC nR ]-sized array of nlsaComponent objects 
%      specifying the source data. nC is the number of components (physical
%      variables) in the dataset and nR the number of realizations (ensemble 
%      members). This is the only mandatory argument in the function call.
%      The remaining arguments are assigned to default values if not set by 
%      the caller. 
%
%   'srcTime': An [ 1 nR ]-sized cell array of vectors specifying the time
%      stamps of each sample in the dataset. The number of elements in 
%      srcTime{ iR } must be equal to the number of samples in 
%      srcComponent( :, iR ). If 'srcTime' is not specified it is set to the
%      default values srcTime{ iR } = 1 : nS( iR ), where nS( iR ) is the 
%      number of samples in the iR-th realization.
% 
%   'tFormat': A string specifying a valid Matlab date format for
%      conversion between numeric and string time data.
%     
%   'embComponent': An [ nC nR ]-sized array of nlsaEmbeddedComponent objects 
%      storing Takens delay-embedded data associacted with 'srcComponent'. The 
%      nlsaEmbeddedComponent class is a child class of nlsaComponent. 
%      two children classes, nlsaEmbeddedComponent_e and 
%      nlsaEmbeddedComponent_o, are derived from nlsaEmbeddedComponent 
%      implementing different storage formats for delay-embedded data. In the 
%      case of nlsaEmbeddedComponent_e objects, the delay embedded data are 
%      are formed and stored in memory explicitly. In the case of 
%      nlsaEmbeddedComponent_o objects, delay embedding is performed on the 
%      fly whenever the data are accessed. The latter format consumes less
%      memory and is generally faster, especially when processing
%      high-dimensnional data. In addition, the classes 
%      nlsaEmbeddedComponent_xi_e and nlsaEmbeddedComponent_xi_o are derived
%      from nlsaEmbaddedComponent_e and nlsaEmbeddedComponent_o. These classes
%      provide storage for the system's phase space velocity estimated 
%      through finite differences.    
%      
%   'trgComponent': An [ nCT nR ]-sized array of nlsaComponent objects 
%      specifying the target data. nCT is the number of target components. 
%      the number of realizations nR must be equal to the number of
%      realizations in the source data. If 'trgComponent' is not
%      specified, it is set equal to 'srcComponent'. 
%
%   'trgTime', 'trgEmbComponent': Same as 'time' and 'embComponent', but for
%      the target data. 
%  
%   Alternatively, the constructor can be called in "template" mode, where 
%   instead of the fully defined objects listed above the arguments supplied
%   by the user only have a set of essential properties defined, and the 
%   remaining properties are filled in automatically. See the class method 
%   parseTemplates for more detais. 
%
%   To manage memory usage and enable certain parallelization features, the 
%   [ nC nR ] and [ nCT nR ]-sized components are furher partioned into 
%   "batches". The batch partitioning is specified through the property 
%   "partition" of  nlsaComponent and nlsaEmbededComponent objects, which is 
%   assigned to an object of class nlsaPartition. Partitions must be mutually 
%   conforming in the sense that 
%   
%     (i) srcComponent( iC, iR ).partition must be identical for fixed 
%     realization (iR) across different components (iC).
%
%     (ii) trgComponent( iCT, iR ).partition must be identical to 
%     srcComponent( iC, iR ) for fixed iR across different ( iC, iCT ).
%
%   Similar conditions apply for the delay-embedded data in embComponent and
%   trgEmbeddedComponent. These partitioning requirements are enforced 
%   automatically when the constructor is called in template mode. 
%
%   Below is a summary of selected methods implemented by this class. These
%   methods can be executed in the sequence listed below.
%
%   - computeDelayEmbedding: Performs Takens delay embedding on the source
%     data in srcComponent; stores the results in embComponent
%
%   - computeTrgDelayEmbedding: Performs delay embedding on the target data
%     in trgComponent; stores the results in trgEmbComponent.
%
%   - computeVelocity: Computes the phase space velocity for the data in 
%     embComponent
%
%   - computeTrgVelocity: Computes the phase space velocity for the data in
%     trgEmbComponent
%
%   These methods can be executed in the sequence listed above. 
%       
%   Contact: dimitris@cims.nyu.edu
%
%   Modified 2018/07/06

    properties
        srcTime         = { 1 };
        trgTime         = { 1 };
        tFormat         = '';
        srcComponent    = nlsaComponent();
        embComponent    = nlsaEmbeddedComponent_e();
        trgComponent    = nlsaComponent();
        trgEmbComponent = nlsaEmbeddedComponent_e();
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaModel_base( varargin )

            msgId = 'nlsa:nlsaModel_base:';

            % Check if constructor is called in "template" mode, and parse
            % templates if needed
            if ifTemplate( 'nlsaModel_base', varargin{ : } )
                varargin = nlsaModel_base.parseTemplates( varargin{ : } );
            end

            nargin   = numel( varargin );

            % Parse input arguments
            iSrcTime         = [];
            iTrgTime         = [];
            iTFormat      = [];
            iSrcComponent    = [];
            iEmbComponent    = [];
            iTrgComponent    = [];
            iTrgEmbComponent = [];
            iTag             = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'srcTime'
                        iSrcTime = i + 1;
                    case 'trgTime'
                        iTrgTime = i + 1;
                    case 'tFormat'
                        iTFormat = i + 1;
                    case 'srcComponent'
                        iSrcComponent = i + 1;
                    case 'embComponent'
                        iEmbComponent = i + 1;
                    case 'trgComponent'
                        iTrgComponent = i + 1;
                    case 'trgEmbComponent'
                        iTrgEmbComponent = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    otherwise
                        error( [ msgId 'invalidProperty' ], ...
                               [ 'Invalid property name ' varargin{ i } ] ) 
                end
            end 

            % Set caller-defined values

            % Source components 
            if ~isempty( iSrcComponent )
                if ~isa( varargin{ iSrcComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidSrc' ], ...
                           'Source data must be specified as an array of nlsaComponent objects.' )
                end
            else
                error( [ msgId 'emptySrc' ], 'Unassigned source components' )
            end
            [ ifC, Test ] = isCompatible( varargin{ iSrcComponent } ); 
            if ~ifC
                disp( Test )
                error( [ msgId 'incompatibleSrc' ], 'Incompatible source component array' )
            end
            obj.srcComponent = varargin{ iSrcComponent };     
            
            [ nC, nR ] = size( obj.srcComponent );
            nD        = getDimension( obj.srcComponent( :, 1 ) ); % dimension of each component
            nSR       = getNSample( obj.srcComponent( 1, : ) );   % number of samples in each realization
           
            % Time for source data
            if ~isempty( iSrcTime )
                if isvector( varargin{ iSrcTime } ) && ~iscell( varargin{ iSrcTime } )
                    obj.srcTime = cell( 1, nR );
                    for iR = 1 : nR
                        obj.srcTime{ iR } = varargin{ iSrcTime };
                    end
                elseif   isvector( varargin{ iSrcTime } ) && iscell( varargin{ iSrcTime } ) ...
                      && numel( varargin{ iSrcTime } ) == nR
                    obj.srcTime = varargin{ iSrcTime };
                else
                    error( [ msgId 'invalidTime' ], ...
                               'Time data must be either a vector or a cell vector of size nR.' )
                end
                if size( obj.srcTime, 1 ) > 1
                    obj.srcTime = obj.srcTime';
                end
                for iR = 1 : nR
                    if ~isvector( obj.srcTime{ iR } ) 
                        msgStr = sprintf( [ 'Invalid timestamp array for realization %i: \n' ... 
                                            'Expecting vector \n' ...
                                            'Received  array of size [%i]' ], ...  
                                            iR, nsR( iR ), size( obj.srcTime{ iR } ) );  
                        error( [ msgId 'invalidTimestamps' ], msgStr )
                    end
                    if numel( obj.srcTime{ iR } ) ~= nSR( iR )
                        msgStr = sprintf( [ 'Invalid number of timestamps for realization %i: \n' ... 
                                            'Expecting %i \n' ...
                                            'Received  %i \n' ], ...  
                                            iR, nsR( iR ), numel( obj.srcTime{ iR } ) );  
                        error( [ msgId 'invalidTimestamps' ], msgStr )
                    end
                    if ~ismonotonic( obj.srcTime{ iR }, 1, 'increasing' )
                        msgStr = sprintf( [ 'Invalid timestamps for realization %i: \n' ... 
                                            'Time values must be monotonically increasing.' ], ...
                                          iR );
                        error( [ msgId 'invalidTimestamps' ], msgStr )
                    end
                    if size( obj.srcTime{ iR }, 1 ) > 1
                        obj.srcTime{ iR } = obj.srcTime{ iR }';
                    end
                end
            else % set to default timestamps if no caller input
                obj.srcTime = cell( 1, nR );
                for iR = 1 : nR
                    obj.srcTime{ iR } = 1 : nSR( iR );
                end
            end

            % Time format
            if ~isempty( iTFormat )
                if ~ischar( varargin{ iTFormat } )
                   error( [ msgId 'tFormat' ], 'Invalid time format' )
                end
                obj.tFormat = varargin{ iTFormat };
            end

            % Embedded components
            % Check input array class and size
            if ~isempty( iEmbComponent )
                if ~isa( varargin{ iEmbComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidEmb' ], ...
                           'Embedded data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
                [ nCE, nRE ] = size( varargin{ iEmbComponent } ); 
                if nCE ~= nC || nRE ~= nR 
                    msgStr = sprintf( [ 'Invalid dimension of embedded component array: \n' ...
                                        'Expecting [%i, %i] \n' ...
                                        'Received  [%i, %i]' ], ...
                                      nC, nR, nCE, nRE );
                    error( [ msgId 'invalidEmb' ], msgStr )
                end
                obj.embComponent = varargin{ iEmbComponent };
            end
            % Check constistency of physical space dimension, embedding indices dimension, and 
            % number of samples
            nSRE = getNSample( obj.embComponent( 1, : ) );
            [ ifC, Test1 ] = isCompatible( obj.embComponent );
            if ~ifC
                msgStr = 'Incompatible embedded component array';
                disp( Test1 )
                error( [ msgId 'incompatibleEmb' ], msgStr )
            end
                        
            % Target components 
            if ~isempty( iTrgComponent )
                if ~isa( varargin{ iTrgComponent }, 'nlsaComponent' )
                    error( [ msgId 'invalidTrg' ], ...
                           'Target data must be specified as an array of nlsaComponent objects.' )
                end
                [ ifC, Test1, Test2 ] = isCompatible( varargin{ iTrgComponent }, ...
                                                      varargin{ iSrcComponent }, ...
                                                      'testComponents', false, ...
                                                      'testSamples', true );
                if ~ifC
                    disp( Test1 )
                    disp( Test2 )
                    error( [ msgId, 'notCompatibleTrg' ], ...
                           'Incompatible target component array' )
                end
                obj.trgComponent = varargin{ iTrgComponent };
  
                [ nCT, nRT ] = size( obj.trgComponent );
                nDT  = getDimension( obj.trgComponent( :, 1 ) ); % dimension of each component
                nSRT = getNSample( obj.trgComponent( 1, : ) );   % number of samples in each realization
           else
                obj.trgComponent = obj.srcComponent;
                nDT              = nD;
                nSRT             = nSR;
                nRT              = nR;
                nCT              = nC;
            end     

            % Time for target data
            if ~isempty( iTrgTime )
                if isvector( varargin{ iTrgTime } ) && ~iscell( varargin{ iTrgTime } )
                    obj.trgTime = cell( 1, nRT );
                    for iR = 1 : nRT
                        obj.trgTime{ iR } = varargin{ iTrgTime };
                    end
                elseif   isvector( varargin{ iTrgTime } ) && iscell( varargin{ iTrgTime } ) ...
                      && numel( varargin{ iTrgTime } ) == nRT
                    obj.trgTime = varargin{ iTrgTime };
                else
                    error( [ msgId 'invalidTrgTime' ], ...
                               'Target time data must be either a vector or a cell vector of size nRT.' )
                end
                if size( obj.trgTime, 1 ) > 1
                    obj.trgTime = obj.trgTime';
                end
                for iR = 1 : nRT
                    if ~isvector( obj.trgTime{ iR } ) 
                        msgStr = sprintf( [ 'Invalid timestamp array for realization %i: \n' ... 
                                            'Expecting vector \n' ...
                                            'Received  array of size [%i]' ], ...  
                                            iR, nsR( iR ), size( obj.trgTime{ iR } ) );  
                        error( [ msg 'invalidTimestamps' ], msgStr )
                    end
                    if numel( obj.trgTime{ iR } ) ~= nSR( iR )
                        msgStr = sprintf( [ 'Invalid number of timestamps for realization %i: \n' ... 
                                            'Expecting %i \n' ...
                                            'Received  %i \n' ], ...  
                                            iR, nsR( iR ), numel( obj.trgTime{ iR } ) );  
                        error( [ msg 'invalidTrgTimestamps' ], msgStr )
                    end
                    if ~ismonotonic( obj.trgTime{ iR }, [], 'increasing' )
                        msgStr = sprintf( [ 'Invalid timestamps for realization %i: \n' ... 
                                            'Time values must be monotonically increasing.' ], ...
                                          iR );
                        error( [ msg 'invalidTrgTimestamps' ], msgStr )
                    end
                    if size( obj.trgTime{ iR }, 1 ) > 1
                        obj.trgTime{ iR } = obj.trgTime{ iR }';
                    end
                end
            else % set to source timestamps if no caller input 
                obj.trgTime = obj.srcTime;
            end

            % Target embedded components
            % Check input array class and size
            if ~isempty( iTrgEmbComponent )
                if ~isa( varargin{ iTrgEmbComponent }, 'nlsaEmbeddedComponent' )
                    error( [ msgId 'invalidTrgEmb' ], ...
                           'Target embedded data must be specified as an array of nlsaEmbeddedComponent objects.' )
                end
                obj.trgEmbComponent = varargin{ iTrgEmbComponent };
            else
                obj.trgEmbComponent = obj.embComponent;
            end
            % Check constistency of physical space dimension, embedding indices dimension, and 
            % number of samples
            [ ifC, Test1, Test2 ] = isCompatible( varargin{ iEmbComponent }, ...
                                                  varargin{ iTrgEmbComponent }, ...
                                                  'testComponents', false, ...
                                                  'testSamples', true );
            if ~ifC
                disp( Test1 )
                disp( Test2 )
                error( [ msgId 'incompatibleSrcTrgEmb' ], ...
                       'Incompatible target embedded component arrays' )
            end                
        end
    end

    methods( Static )
 
        %% LISTCONSTRUCTORPROPERTIES List property names for class constructor
        function pNames = listConstructorProperties
            pNames = { 'srcComponent' ...
                       'srcTime' ...
                       'tFormat' ...
                       'embComponent' ...
                       'trgComponent' ...
                       'trgTime' ...
                       'trgEmbComponent' };
        end
 
        %% LISTPARSERPROPERTIES  List property names for class constructor parser
        function pNames = listParserProperties
            pNames = { 'sourceComponent' ... 
                       'sourceTime' ...
                       'timeFormat' ...
                       'embeddingOrigin' ...
                       'embeddingTemplate' ...
                       'embeddingPartition' ...
                       'targetComponent' ...
                       'targetEmbeddingTemplate' ...
                       'targetEmbeddingOrigin' };
        end

        %% PARSETEMPLATES  Template parser
        propNameVal = parseTemplates( varargin );
   end
end
