classdef nlsaKoopmanOperator < nlsaKernelOperator
%NLSAKOOPMANOPERATOR Class definition and constructor of regularized Koopman 
% operators 
% 
% Modified 2020/04/08    

    %% PROPERTIES
    properties
        epsilon    = 1;                % regularization parameter 
        dt         = 1;                % sampling interval
        fileLambda = 'dataLambda.mat'; % eigenvalues
        fileE      = 'dataE.mat';      % Dirichlet energies
        pathV      = 'dataV';          % path for operator storage
        pathPhi    = 'dataPhi';        % path for eigenvalues/eigenfunctions 
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKoopmanOperator( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iEpsilon       = [];
            iDt            = [];
            iFileLambda    = [];
            ifileE         = [];
            iPathP         = [];
            iPathPhi       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'epsilon'
                        iEpsilon = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'samplingInterval'
                        iDt = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'eigenvalueFile'
                        iFileLambda = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'energyFile'
                        iFileE = i + 1;
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
            if ~isempty( iEpsilon )
                if ~isps( varargin{ iEpsilon } )
                    msgStr = [ 'The regularization parameter must be a  ' ...
                               'positive scalar' ];
                    error( msgStr )
                end
                obj.epsilon = varargin{ iEpsilon };
            end
            if ~isempty( iDt )
                if ~isps( varargin{ iDt } )
                    msgStr = [ 'The sampling interval must be a  ' ...
                               'positive scalar' ];
                    error( msgStr )
                end
                obj.dt = varargin{ iDt };
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
