classdef nlsaDiffusionOperator < nlsaKernelOperator
%NLSADIFFUSIONOPERATOR Class definition and constructor of diffusion 
%  operator, implementing the diffusion maps algorithm in
% 
%  R. R. Coifman and S. Lafon (2006), "Diffusion Maps", Appl. 
%  Comput. Harmon. Anal., 21, 5, doi:10.1016/j.acha.2006.04.006
%  
% The following properties are defined:
%
% 'mode': A string specifying wether eigendecomposition or singular 
%    value decomposition is to be perfomed. mode = 'eig' performs 
%    eigendecomposition. mode = 'svd' performs SVD, saving the 
%    left singular vectors and the squares of the eigenvalues. This is
%    equivalent to computing the eigenvalues and eigenvectors of the suymmetric
%    matrix PP^*.
%
%
% Modified 2018/06/10    

    %% PROPERTIES
    properties
        alpha      = 0;
        epsilon    = 1;
        epsilonT   = []; % reverts to epsilon if empty
        fileLambda = 'dataLambda.mat';
        pathP      = 'dataP';
        pathPhi    = 'dataPhi';
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaDiffusionOperator( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iAlpha         = [];
            iEpsilon       = [];
            iEpsilonT      = []; 
            iFileLambda    = [];
            iPathP         = [];
            iPathPhi       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'alpha'
                        iAlpha = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'epsilon'
                        iEpsilon = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'epsilonT'
                        iEpsilonT = i + 1;
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

            % Set caller defined values
            if ~isempty( iAlpha )
                if ~isrs( varargin{ iAlpha } )
                    error( 'Normalization parameter alpha must be a real scalar' )
                end
                obj.alpha = varargin{ iAlpha };
            end

            % Set caller-defined values
            if ~isempty( iEpsilon )
                if ~isps( varargin{ iEpsilon } )
                    error( 'The bandwidth parameter must be a positive scalar' )
                end
                obj.epsilon = varargin{ iEpsilon };
            end
            if ~isempty( iEpsilonT )
                if ~isps( varargin{ iEpsilonT } )
                    error( 'The bandwidth parameter for the test data must be a positive scalar' )
                end
                obj.epsilonT = varargin{ iEpsilonT };
            end
            if ~isempty( iFileLambda )
                if ~isrowstr( varargin{ iFileLambda } )
                    error( 'Invalid eigenvalue file specification' )
                end
                obj.fileLambda = varargin{ iFileLambda };
            end
            if ~isempty( iPathP )
                if ~isrowstr( varargin{ iPathP } )
                    error( 'Invalid operator subpath specification' )
                end
                obj.pathP = varargin{ iPathP };
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
