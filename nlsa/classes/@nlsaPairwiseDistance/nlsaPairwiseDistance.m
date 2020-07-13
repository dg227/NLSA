classdef nlsaPairwiseDistance
%NLSAPAIRWISEDISTANCE  Class definition and constructor of nlsaPairwiseDistance
% objects
%
% Modified 2020/07/13  

   %% PROPERTIES
    properties
        file          = nlsaFilelist();
        dFunc         = nlsaLocalDistanceFunction();
        nN            = 2;
        partition     = nlsaPartition();  
        partitionT    = nlsaPartition.empty(); % reverts to partition if empty
        path          = pwd;
        pathY         = 'dataY';
        tag           = '';
        nPar          = 0;
    end

    methods

        %% NLSAPAIRWISEDISTANCE  Class contructor
        function obj = nlsaPairwiseDistance( varargin )

            % Parse input arguments
            iFile          = [];
            iDFunc         = [];
            iNN            = [];
            iPartition     = [];
            iPartitionT    = [];
            iPath          = [];
            iPathY         = [];
            iTag           = [];
            iNPar          = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'file'
                        iFile = i + 1;
                    case 'distanceFunction'
                        iDFunc = i + 1;
                    case 'nearestNeighbors'
                        iNN = i + 1;
                    case 'partition'
                        iPartition = i + 1;
                    case 'partitionT'
                        iPartitionT = i + 1;
                    case 'path'
                        iPath = i + 1;
                    case 'distanceSubpath'
                        iPathY = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    case 'nPar'
                        iNPar = i + 1;
                    otherwise
                        error( 'Invalid property' )
                end
            end
            % Set caller defined values
           if ~isempty( iPartition )
               if ~isa( varargin{ iPartition }, 'nlsaPartition' ) ...
                  || ~isrow( varargin{ iPartition } )
                   error( 'Partition property must be set to a row vector of nlsaPartition objects' )
               end
               obj.partition = varargin{ iPartition };
            end
            if ~isempty( iPartitionT )
               if ~isa( varargin{ iPartitionT }, 'nlsaPartition' ) ...
                   || ~isrow( varargin{ iPartitionT } )
                   error( 'Test partition property must be set to a row vector of nlsaPartition objects' )
               end
               obj.partitionT = varargin{ iPartitionT };
            end                
            if ~isempty( iFile )
                if ~isa( varargin{ iFile }, 'nlsaFilelist' ) ...
                   || ~iscompatible( varargin{ iFile }, obj.partition )                        
                     error( 'File property must be set to a vector of nlsaFilelist objects compatible with the partition' )
                end
                obj.file = varargin{ iFile };
            else
                obj.file = nlsaFilelist( obj.partition );
            end
            if ~isempty( iDFunc )
                if ~isa( varargin{ iDFunc }, 'nlsaLocalDistanceFunction' )
                    error( 'Local distance function must be set to an nlsaLocalDistanceFunction object' )
                end
                obj.dFunc = varargin{ iDFunc };
            end
            if ~isempty( iNN )
                if ~ispsi( varargin{ iNN } )
                    error( 'Number of nearest neighbors must be a positive scalar integer' )
                end
                obj.nN = varargin{ iNN };
            end
            if ~isempty( iPath )
                if ~isrowstr( varargin{ iPath } )
                    error( 'Invalid path specification' )
                end
                obj.path = varargin{ iPath }; 
            end
            if ~isempty( iPathY )
                if ~isrowstr( varargin{ iPathY } )
                    error( 'Invalid subpath specification' )
                end 
                obj.pathY = varargin{ iPathY }; 
            end 
            if ~isempty( iTag )
                if iscell( varargin{ iTag } ) || isrowstr( varargin{ iTag } ) 
                    obj.tag = varargin{ iTag };
                else
                    error( 'Invalid object tag' )
                end
            end
            if ~isempty( iNPar )
                if ~isnnsi( varargin{ iNPar } )
                    msgStr = [ 'Number of parallel workers must be a ' ...
                               'nonnegative scalar integer.' ]; 
                    error( msgStr )
                end
                obj.nPar = varargin{ iNPar };
            end
        end
    end
end    
