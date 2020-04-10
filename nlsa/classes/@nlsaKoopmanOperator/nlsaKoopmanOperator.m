classdef nlsaKoopmanOperator < nlsaKernelOperator
%NLSAKOOPMANOPERATOR Class definition and constructor of Koopman/Perron-
% Frobenius generators based on temporal finite differences.
% 
% Modified 2020/04/09    

    %% PROPERTIES
    properties
        dt         = 1;         % sampling interval
        fdOrd      = 2;         % finite-difference order
        fdType     = 'central'; % finite-difference type: backward 
                                %                         forward  
                                %                         central  
        fileLambda = 'dataLambda.mat'; % eigenvalues
        pathV      = 'dataV';          % path for operator storage
        pathPhi    = 'dataPhi';        % path for eigenvalues/eigenfunctions 
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKoopmanOperator( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iDt            = [];
            iFdOrd         = [];
            iFdType        = [];
            iFileLambda    = [];
            iPathP         = [];
            iPathPhi       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'samplingInterval'
                        iDt = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fdOrder'
                        iFdOrd = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'fdType'
                        iFdType = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenvalueFile'
                        iFileLambda = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'operatorSubpath'
                        iPathP = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenfunctionSubpath'
                        iPathPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'tag'
                        iTag = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaKernelOperator( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iDt )
                if ~isps( varargin{ iDt } )
                    msgStr = [ 'The sampling interval must be a  ' ...
                               'positive scalar' ];
                    error( msgStr )
                end
                obj.dt = varargin{ iDt };
            end
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
            if ~isempty( iFileLambda )
                if ~isrowstr( varargin{ iFileLambda } )
                    error( 'Invalid eigenvalue file specification' )
                end
                obj.fileLambda = varargin{ iFileLambda };
            end
            if ~isempty( iFilewEambda )
                if ~isrowstr( varargin{ iFileE } )
                    error( 'Invalid energy file specification' )
                end
                obj.fileE = varargin{ iFileE };
            end
            if ~isempty( iPathP )
                if ~isrowstr( varargin{ iPathP } )
                    error( 'Invalid operator subpath specification' )
                end
                obj.pathV = varargin{ iPathP };
            end
            if ~isempty( iPathPhi )
                if ~isrowstr( varargin{ iPathPhi } )
                    error( 'Invalid eigenfunction subpath specification' )
                end
                obj.pathPhi = varargin{ iPathPhi };
            end
        end
    end

end    
