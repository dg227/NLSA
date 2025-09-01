classdef nlsaSymmetricDistance
%NLSASYMMETRICDISTANCE Class definition and constructor of NLSA symmetric distance matrix
%
% Modified 2014/04/11    

    properties
        nN            = 2;
        partition     = nlsaPartition();
        path          = pwd;
        pathYS        = 'dataYS';
        tag           = '';
    end

    methods

        function obj = nlsaSymmetricDistance( varargin )
            % Return default object if no arguments
            if nargin == 0
                return
            end
            % Parse input arguments
            iFile          = [];
            iNN            = [];
            iPartition     = [];
            iPath          = [];
            iPathYS        = [];
            iTag           = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'file'
                        iFile = i + 1;
                    case 'nearestNeighbors'
                        iNN = i + 1;
                    case 'partition'
                        iPartition = i + 1;
                    case 'path'
                        iPath = i + 1;
                    case 'distanceSubpath'
                        iPathYS = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    otherwise
                        error( 'Invalid property' )
                end
            end
            % Set caller defined values
            if ~isempty( iNN )
                if ~ispsi( varargin{ iNN } )
                    error( 'Number of nearest neighbors must be a positive scalar integer' )
                end
                obj.nN = varargin{ iNN };
            end
            if ~isempty( iPartition )
                if ~isa( varargin{ iPath }, 'nlsaPartition' ) ...
                  || ~isrow( varargin{ iPath } )
                    error( 'Partition property must be set to a row vector of nlsaPartition objects' )
                end
            end
            if ~isempty( iPath )
                if ~isrowstr( varargin{ iPath } )
                    error( 'Invalid path specification' )
                end
                obj.path = varargin{ iPath }; 
            end
            if ~isempty( iPathYS )
                if ~isrowstr( varargin{ iPathYS } )
                    error( 'Invalid distance subpath specification' )
                end
                obj.pathYS = varargin{ iPathYS }; 
            end
            if ~isempty( iTag )
                if ~iscell( varargin{ iTag } ) ...
                  || ~isrowstr( varargin{ iTag } ) 
                    error( 'Invalid object tag' )
                end
                obj.tag = varargin{ iTag };
            end
        end
    end

    methods( Abstract )

        %% SYMMETRIZEDISTANCES Perform distance symmetrization
        symmetrizeDistances( obj )

        %% GETDISTANCES Read distance data
        y = getDistances( obj )
        
        %% SETDISTANCES Write distance data
        setDistances( obj, y )
    end

end    
