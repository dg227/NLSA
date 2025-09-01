classdef nlsaProjectedComponent 
%NLSAPROJECTEDCOMPONENT  Class definition and constructor of data components
% projected onto an eigenfunction basis
%
% Modified 2014/06/23


    properties
        nL        = 1;                % Number of basis functions
        nDE       = 1;                % Embedding space dimension 
        partition = nlsaPartition();  % Vector specifying the number of samples in each realization
        fileA    = 'dataA.mat';       % projected data
        path     = pwd;
        pathA    = pwd;
        tag      = '';
    end

    methods

        function obj = nlsaProjectedComponent( varargin )
            % Return default object if no arguments
            if nargin == 0
                return
            end
            % Parse input arguments
            iNL            = [];
            iFileA         = [];
            iNDE           = [];
            iPartition    = [];
            iPath          = [];
            iPathA         = [];
            iTag           = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'nBasisFunction'
                        iNL = i + 1;
                    case 'partition'
                        iPartition = i + 1;
                    case 'embeddingSpaceDimension'
                        iNDE = i + 1;
                    case 'projectionFile'
                        iFileA = i + 1;
                    case 'path'
                        iPath = i + 1;
                    case 'projectionSubpath'
                        iPathA = i + 1;
                    case 'tag'
                        iTag = i + 1;
                    otherwise
                        error( [ 'Invalid property ', varargin{ i } ] )
                end
            end

            % Set caller defined values
            if ~isempty( iNL )
                if ~ispsi( varargin{ iNL } ) 
                        error( 'Number of basis functions must be a positive scalar integer' )
                end
                obj.nL = varargin{ iNL };
            end
            if ~isempty( iPartition )                                
                if ~isa( varargin{ iPartition }, 'nlsaPartition' ) ...
                  || ~isrow( varargin{ iPartition } )
                   error( 'Partition property must be set to a row vector of nlsaPartition objects' )
                end
                obj.partition = varargin{ iPartition };
            end
            if ~isempty( iNDE )
                if ~ispsi( varargin{ iNDE } )
                    error( 'Embedding space dimension must be a positive scalar integer' )
                end
                obj.nDE = varargin{ iNDE };
            end
            if ~isempty( iFileA )
                if ~isrowstr( varargin{ iFileA } )
                    error( 'FileA property must be set to a string' )
                end
                obj.fileA = varargin{ iFileA };
            end
            if ~isempty( iPath )
                if ~isrowstr( varargin{ iPath } )
                    error( 'Invalid path specification' )
                end
                obj.path = varargin{ iPath }; 
            end
            if ~isempty( iPathA )
                if ~isrowstr( varargin{ iPathA } )
                    error( 'Invalid state projection path specification' ) 
                end
                obj.pathA = varargin{ iPathA };
            end
            if ~isempty( iTag )
                if iscell( varargin{ iTag } ) || isrowstr( varargin{ iTag } ) 
                    obj.tag = varargin{ iTag };
                else
                    error( 'Invalid object tag' )
                end
            end
        end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaProjectedComponent';
        end
    end
end    
