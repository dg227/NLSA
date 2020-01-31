classdef nlsaComponent < nlsaRoot
%%NLSACOMPONENT  Class definition and constructor of NLSA data component
%
% Modified 2019/11/13    

    properties
        nD        = 1;                         % dimension
        partition = nlsaPartition();
        path      = pwd;
        pathX     = '.';
        file      = nlsaFilelist();
        tagC      = '';    % component tag
        tagR      = '';    % realization tag
    end

    methods


        %%CLASS CONSTRUCTOR
        function obj = nlsaComponent( varargin )
            % Return default object if no arguments
            if nargin == 0
                return
            end

            % Parse input arguments
            iD          = [];
            iFile       = [];
            iPartition  = [];
            iPath       = [];
            iPathX      = [];            
            iTagC       = [];
            iTagR       = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'dimension'
                        iD = i + 1;
                    case 'file'
                        iFile = i + 1;
                    case 'partition'
                        iPartition = i + 1;
                    case 'path'
                        iPath = i + 1;
                    case 'pathX'
                        iPathX = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'componentTag'
                        iTagC = i + 1;
                    case 'realizationTag'
                        iTagR = i + 1;
                    otherwise
                        error( [ 'Invalid property ' varargin{ i } ] )
                end
            end
            % Set caller defined values
            if ~isempty( iPartition )
               if isa( varargin{ iPartition }, 'nlsaPartition' )
                   obj.partition = varargin{ iPartition };
               else
                   error( 'Partition property must be set to a nlsaPartion object' )
               end
            end
            nB  = getNBatch( obj.partition );
            if ~isempty( iD )
                if ~ispsi( varargin{ iD } )
                    error( 'Invalid component dimension' )
                end
                obj.nD = varargin{ iD };
            end
            if ~isempty( iPath )
                if ~isrowstr( varargin{ iPath } )
                    error( 'Invalid path specification' )
                end
                obj.path = varargin{ iPath }; 
            end
            if ~isempty( iPathX )
                if ~isrowstr( varargin{ iPathX } )
                    error( 'Invalid data subdirectory' )
                end
                obj.path = varargin{ iPathX }; 
            end
            if ~isempty( iFile )
                if ~isa( varargin{ iFile }, 'nlsaFilelist' ) ...
                  || ~isscalar( varargin{ iFile } ) ...
                  || getNFile( varargin{ iFile } ) ~= getNBatch( obj.partition )
                    error( 'File property must be set to an nlsaFilelist object with number of files equal to the number of batches' )
                end
                obj.file = varargin{ iFile };
            else
                obj.file = nlsaFilelist( 'nFile', getNBatch( obj.partition )  );
            end
            if ~isempty( iTagC )
                if iscell( varargin{ iTagC } ) || isrowstr( varargin{ iTagC } ) 
                    obj.tagC = varargin{ iTagC };
                else
                    error( 'Invalid component tag' )
                end
            end
            if ~isempty( iTagR )
                if iscell( varargin{ iTagC } ) || isrowstr( varargin{ iTagR } ) 
                    obj.tagR = varargin{ iTagR };
                else
                    error( 'Invalid realization tag' )
                end
            end
        end
    
        %% SET/GET METHODS THAT REQUIRE SPECIFIC IMPLEMENTATION
        function obj = set.partition( obj, partition )
            % Validate input arguments
            if ~isa( partition, 'nlsaPartition' ) || ~isscalar( partition )
                error( 'Partition must be a scalar nlsaPartition object' )
            end

            % Quick return if the partition to be assigned is the same as the 
            % existing partition
            if isequal( obj.partition, partition )
                return
            end

            obj.partition = partition;
            % Reset fileList since partition has changed 
            obj.file = nlsaFilelist( 'nFile', getNBatch( partition ) );

            obj = setPartition( obj, partition );
        end

    end
end    
