classdef nlsaEmbeddedComponent_xi < nlsaEmbeddedComponent
%NLSAEMBEDDEDCOMPONENT_XI  Class definition and constructor of NLSA 
% time-lagged embedded component with phase space velocity data
%
% Modified 2020/04/09    

    properties
        fdOrd         = 1;          % finite-difference order
        fdType        = 'backward'; % finite-difference type: backward 
                                    %                         forward  
                                    %                         central  
        fileXi        = nlsaFilelist();
        pathXi        = 'dataXi';
        tagXi         = ''; % velocity tag
    end

    methods

        function obj = nlsaEmbeddedComponent_xi( varargin )

            msgId = 'nlsa:nlsaEmbeddedComponent_xi';
            
            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iFdOrd      = [];
            iFdType     = [];
            iFileXi     = [];
            iPathXi     = [];
            iTagXi      = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'pathXi'
                        iPathXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fileXi'
                        iFileXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fdOrder'
                        iFdOrd = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fdType'
                        iFdType = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'velocityTag'
                        iTagXi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaEmbeddedComponent( varargin{ ifParentArg } );
            
            % Set caller-defined values
            if ~isempty( iFdOrd )
                if ~ispsi( varargin{ iFdOrd } )
                    msgStr = [ 'Finite-difference order must be a ' ... 
                               'positive scalar integer' ] 
                    error( msgStr )
                end
                obj.fdOrd = varargin{ iFdOrd };
            end
            if ~isempty( iFdType )
                if ischar( varargin{ iFdType } )
                    if any( strcmp( varargin{ iFdType }, ...
                        { 'forward' 'backward' 'central' } ) )  
                        ifErr      = false;
                        obj.fdType = varargin{ iFdType };
                    else
                        ifErr = true;
                    end
                else
                    ifErr = true;
                end
                if ifErr
                    error( 'Invalid specification of finite-difference type' )
                end
                switch obj.fdType
                case 'central'
                    if ~iseven( obj.fdOrd )
                        msgStr = [ 'Central finite-difference schemes ' ...
                                   'must be of even order' ]; 
                        error( msgStr )
                    end
                otherwise
                    if ~isodd( obj.fdOrd )
                        msgStr = [ 'Forward and backward finite-difference '...
                                   'schemes must be of odd order' ];
                        error( msgStr )
                    end
                end
            end
            if ~isempty( iPathXi )
                if ~ischar( varargin{ iPathXi } )
                    error( 'Invalid path specification' )
                end
                obj.pathXi = varargin{ iPathXi };
            end
            if ~isempty( iFileXi )
                 if ~isa( varargin{ iFileXi }, 'nlsaFilelist' ) ...
                   || ~isscalar( varargin{ iFileXi } ) ...
                   || getNFile( varargin{ iFileXi } ) ~= ...
                       getNBatch( obj.partition )
                     msgStr = [ 'FileXi property must be set to an ' ...
                                'nlsaFilelist object with number of files ' ...
                                'equal to the number of batches' ];
                     error( msgStr )
                 end
                 obj.fileXi = varargin{ iFileXi };
            else
                obj.fileXi = nlsaFilelist( ...
                    'nFile', getNBatch( obj.partition ) );
            end
            if ~isempty( iTagXi )
                if ~isrowstr( varargin{ iTagXi } ) 
                    error( 'Invalid velocity tag' )
                end
                obj.tagXi = varargin{ iTagXi };
            end 
        end        
    end

    methods( Abstract )
        %% GETVELOCITY Returns the phase space velocity and its squared norm
        [ xiNorm2, xi ] = getVelocity( obj, outFormat );

        %% GETVELOCITYNORM Returns the phase space velocity norm
        xiNorm = getVelocityNorm( obj, outFormat )
    end
end    
