classdef nlsaEmbeddedComponent_ose < nlsaEmbeddedComponent_xi_e
%NLSAEMBEDDEDCOMPONENT_OSE  Class definition and constructor of NLSA 
% time-lagged embedded component with out-of-sample extension for phase
% space velocity data
%
% Modified 2015/12/14

    properties
        fileEX   = nlsaFilelist(); % state OSE error
        fileEXi = nlsaFilelist();  % phase space velocity OSE error  
        pathEX  = 'dataEX';        % state error subdirectory 
        pathEXi = 'dataEXi';       % velocity error subdirectory 
        tagO    = '';              % OSE tag
    end

    methods

    %% CLASS CONSTRUCTOR
        function obj = nlsaEmbeddedComponent_ose( varargin )

            nargin = numel( varargin );

            if nargin == 1 && isa( varargin{ 1 }, 'nlsaEmbeddedComponent_xi' ) 
                varargin = { 'dimension', getDimension( varargin{ 1 } ), ...
                             'partition', getPartition( varargin{ 1 } ), ...
                             'origin', getOrigin( varargin{ 1 } ), ...
                             'idxE', getEmbeddingIndices( varargin{ 1 } ), ...
                             'nXB', getNXB( varargin{ 1 } ), ...
                             'nXA', getNXA( varargin{ 1 } ), ...
                             'fdOrder', getFDOrder( varargin{ 1 } ), ...
                             'fdType', getFDType( varargin{ 1 } ), ...
                             'componentTag', getComponentTag( varargin{ 1 } ), ...
                             'realizationTag', getRealizationTag( varargin{ 1 } ) };

                nargin = numel( varargin );
            end
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFileEX  = [];
            iFileEXi = [];  
            iPathEXi = [];
            iPathEX  = [];
            iTagO    = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'fileX'
                        iFileEX = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fileEXi'
                        iFileEXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'pathEX'
                        iPathEX = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'pathEXi'
                        iPathEXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'oseTag'
                        iTagO = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaEmbeddedComponent_xi_e( varargin{ ifParentArg } );

                        
            % Set caller-defined values
            if ~isempty( iFileEX )
                if ~isa( varargin{ iFileEX }, 'nlsaFilelist' ) ...
                  || ~isscalar( varargin{ iFileEX } ) ...
                  || getNFile( varargin{ iFileEX } ) ~= getNBatch( obj.partition )
                    error( 'FileEX property must be set to an nlsaFilelist object with number of files equal to the number of batches' )
                end
                obj.fileEX = varargin{ iFileEX };
            else
                obj.fileEX = nlsaFilelist( 'nFile', getNBatch( obj.partition ) );
            end
            if ~isempty( iFileEXi )
                if ~isa( varargin{ iFileEXi }, 'nlsaFilelist' ) ...
                  || ~isscalar( varargin{ iFileEXi } ) ...
                  || getNFile( varargin{ iFileEXi } ) ~= getNBatch( obj.partition )
                    error( 'FileEXi property must be set to an nlsaFilelist object with number of files equal to the number of batches' )
                end
                obj.fileEXi = varargin{ iFileEXi };
            else
                obj.fileEXi = nlsaFilelist( 'nFile', getNBatch( obj.partition ) );
            end
            if ~isempty( iPathEX )
                if isrowstr( varargin{ iPathEX } )
                    obj.pathEX = varargin{ iPathEX };
                else
                    error( 'Invalid state error path specification' ) 
                end
            end
            if ~isempty( iPathEXi )
                if isrowstr( varargin{ iPathEXi } )
                    obj.pathEXi = varargin{ iPathEXi };
                else
                    error( 'Invalid velocity error path specification' ) 
                end
            end
            if ~isempty( iTagO )
                if ~isrowstr( varargin{ iTagO } )
                    error( 'Invalid OSE tag' )
                end
                obj.tagO = varargin{ iTagO };
            end

        end                    
    end

    methods( Static )

        %% GETERRORMSGID  Default error message ID for class
        function mId = getErrMsgId
            mId = 'nlsa:nlsaEmbeddedComponent_ose';
        end

    end
end    
