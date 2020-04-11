classdef nlsaKoopmanOperator < nlsaKernelOperator
%NLSAKOOPMANOPERATOR Class definition and constructor of Koopman/Perron-
% Frobenius generators based on temporal finite differences.
% 
% Modified 2020/04/11    

    %% PROPERTIES
    properties
        dt         = 1;         % sampling interval
        fdOrd      = 2;         % finite-difference order
        fdType     = 'central'; % finite-difference type: backward 
                                %                         forward  
                                %                         central  
        antisym    = false;            % forces antisymmetric matrix if true
        idxBasis   = 1;                % basis function indices
        fileEVal   = 'dataGamma.mat';  % eigenvalues
        fileEFunc  = 'dataZeta.mat';   % eigenfunctions
        fileCoeff  = 'dataC.mat';      % eigenfunction expansion coefficients 
        fileOp     = 'dataV.mat';      % operator file
        pathOp     = 'dataV';          % path for operator storage
        pathEig    = 'dataPhi';        % path for eigenvalues/eigenfunctions 
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKoopmanOperator( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iDt            = [];
            iFdOrd         = [];
            iFdType        = [];
            iIdxPhi        = [];
            iAntisym       = [];
            iFileEVal      = [];
            iFileEFunc     = [];
            iFileCoeff     = [];
            iFileOp        = [];
            iPathOp        = [];
            iPathEig       = [];

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
                    case 'iAntisym'
                        iAntisym = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'basisFunctionIdx'
                        iIdxPhi = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenvalueFile'
                        iFileEVal = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenvalueFile'
                        iFileEVal = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenfunctionFile'
                        iFileEFunc = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'coefficientFile'
                        iFileCoeff = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'operatorFile'
                        iFileOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'operatorSubpath'
                        iPathOp = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenfunctionSubpath'
                        iPathEig = i + 1;
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
            if ~isempty( iAntisym )
                if ~isscalar( varargin{ iAntisym } ) ...
                    || ~islogical( varargin{ iAntisym } )
                    msgStr = [ 'Antisymmetrization property must be ' ...
                               'specified as a logical scalar' ];
                    error( msgStr )
                end
                obj.antisym = varargin{ iAntisym };
            end 
            if ~isempty( iIdxPhi ) 
                if ~isempty( iNeig )
                    msgStr = [ 'Eigenfunction indices and number of ' ...
                               'eigenfunctions cannot be specified ' ...
                               'simultaneously' ];
                    error( msgStr  )
                end
                if ~obj.isValidIdx( varargin{ iIdxPhi } ) 
                    error( 'Invalid basis function index specification' )
                end
                obj.idxBasis = varargin{ iIdxPhi };
                obj = setNEigenfunction( obj, numel( varargin{ iIdxPhi } ) );
            else 
                obj.idxBasis = 1 : getNEigenfunction( obj );
            end
            if ~isempty( iFileEVal )
                if ~isrowstr( varargin{ iFileEVal } )
                    error( 'Invalid eigenvalue file specification' )
                end
                obj.fileEVal = varargin{ iFileEVal };
            end
            if ~isempty( iFileEFunc )
                if ~isrowstr( varargin{ iFileEFunc } )
                    error( 'Invalid coefficient file specification' )
                end
                obj.fileEFunc = varargin{ iFileEFunc };
            end
            if ~isempty( iFileCoeff )
                if ~isrowstr( varargin{ iFileCoeff } )
                    error( 'Invalid coefficient file specification' )
                end
                obj.fileCoeff = varargin{ iFileCoeff };
            end
            if ~isempty( iFileOp )
                if ~isrowstr( varargin{ iFileOp } )
                    error( 'Invalid operator file specification' )
                end
                obj.fileOp = varargin{ iFileOp };
            end
            if ~isempty( iPathOp )
                if ~isrowstr( varargin{ iPathOp } )
                    error( 'Invalid operator subpath specification' )
                end
                obj.pathOp = varargin{ iPathOp };
            end
            if ~isempty( iPathEig )
                if ~isrowstr( varargin{ iPathEig } )
                    error( 'Invalid eigenfunction subpath specification' )
                end
                obj.pathEig = varargin{ iPathEig };
            end
        end
    end

end    
