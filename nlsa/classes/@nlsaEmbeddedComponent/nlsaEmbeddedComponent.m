classdef nlsaEmbeddedComponent < nlsaComponent
%NLSAEMBEDDEDCOMPONENT  Class definition and constructor of NLSA time-lagged embedded component
%
% Modified 2015/08/31

    properties
        idxO          = 1;      % time origin
        idxE          = 1;      % embedding indices
        nXB           = 0;      % number of samples before the main time interval (for phase space velocity)
        nXA           = 0;      % number of samples after the main time interval (for phase space velocity)
        fileB         = 'dataXB.mat';
        fileA         = 'dataXA.mat';
        tagE          = ''; % embedding tag
    end

    methods

        function obj = nlsaEmbeddedComponent( varargin )

            msgId = 'nlsa:nlsaEmbeddedComponent';
            
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iIdxE       = [];
            iIdxO       = [];
            iNXB        = [];
            iNXA        = [];
            iFileB      = [];
            iFileA      = [];
            iNE         = [];
            iNESkip     = [];
            iTagE       = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'fileB'
                        iFileB = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fileA'
                        iFileA = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'idxE'
                        iIdxE = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'origin'
                        iIdxO = i + 1; 
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nEmbed'
                        iNE = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nEmbedSkip'
                        iNESkip = i + 1; 
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nXB'
                        iNXB = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'nXA'
                        iNXA = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'embeddingTag';
                        iTagE = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaComponent( varargin{ ifParentArg } );
            
            % Set caller-defined values
            nB  = getNBatch( obj.partition );
            if ~isempty( iIdxE )
                if isempty( iNE ) && isempty( iNESkip )
                    idxE = varargin{ iIdxE };
                    if ~isvector( idxE )
                        error( 'Embedding indices must be passed in a vector' )
                    end
                    if size( idxE, 1 ) > 1
                        idxE = idxE';
                    end
                    nSE = numel( idxE );
                    idxRef = 0;
                    for iSE = 1 : nSE
                        if ~ispsi( idxE( iSE ) - idxRef )
                            error( 'Invalid index specification' )
                        end
                        idxRef = idxE( iSE );
                    end
                    obj.idxE = idxE;
                else
                    error( 'Explicit embedding indices cannot be specified if nE and/or nESkip are specified' )
                end 
                obj.idxE = varargin{ iIdxE };
            else
                if ~isempty( iNE )
                    nE = varargin{ iNE };
                    if ~ispsi( nE ) 
                        error( 'Number of embedding snapshots must be a positive integer' )
                    end
                else
                    nE = 1;
                end
                if ~isempty( iNESkip )
                    nESkip = varargin{ iNESkip };
                    if ~isnzsi( nE - nESkip )
                        error( 'Sampling interval must not exceed length of lagged embedding window' )
                    end
                else
                    nESkip = 1;
                end
                obj.idxE = 1 : nESkip : nE;
            end
            if ~isempty( iIdxO )
                if ~ispsi( varargin{ iIdxO } )
                    error( 'Time origin for embedding must be a positive scalar integer' )
                end
                obj.idxO = varargin{ iIdxO };
            end
            if ~isempty( iFileB )
                if ~isrowstr( varargin{ iFileB } )
                    error( 'Invalid specification of data files' )
                end
                obj.fileB = varargin{ iFileB };
            end
            if ~isempty( iFileA )
                if ~ischar( varargin{ iFileA } )
                    error( 'Invalid specification of data files' )
                end
                obj.fileA = varargin{ iFileA };
            end
            if ~isempty( iNXB )
                if ~isnnsi( varargin{ iNXB } )
                    error( 'Number of before samples must be a non-negative integer' )
                end
                obj.nXB = varargin{ iNXB };
            end 
            if ~isempty( iNXA )
                if ~isnnsi( varargin{ iNXA } )
                    error( 'Number of after samples must be a non-negative integer' )
                end
                obj.nXA = varargin{ iNXA };
            end 
            if ~isempty( iTagE )
                if ~isrowstr( varargin{ iTagE } ) 
                    error( 'Invalid embedding tag' )
                end
                obj.tagE = varargin{ iTagE };
            end 
        end
    end

    methods( Abstract )

        %% GETBATCHARRAYSIZE Return the size of data arrays 
        [ nDE, nSB ] = getBatchArraySize( obj )
    
        %% GETDATA Read batch data
        x = getData( obj, outFormat )

        %% GETEXTRABATCHSIZE Get the number of "extra" samples in the data 
        % arrays due to embedding
        nSBE = getExtraBatchSize( obj )

        %% computeData Perform time-lagged embedding
        computeData( obj )

        %% NORM2 Compute quared norm of data 
        x2 = norm2( obj, x )
    end

end    
