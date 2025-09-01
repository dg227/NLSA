classdef nlsaKoopmanOperator_diff < nlsaKoopmanOperator
%NLSAKOOPMANOPERATOR_DIFF Class definition and constructor of Koopman generator
% with diffusion regularization.
% 
% Modified 2020/04/08    

    %% PROPERTIES
    properties
        epsilon = 1;     % regularization parameter 
        regType = 'lin'; % regularization type: lin (linear)  
                         %                      log (logarithmic)
                         %                      inv (inverse)
    end

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaKoopmanOperator_diff( varargin )

            ifParentArg = true( 1, nargin );
 
            % Parse input arguments
            iEpsilon       = [];
            iRegType       = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'regularizationParameter'
                        iEpsilon = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'regularizationType'
                        iRegType = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaKoopmanOperator( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iEpsilon )
                if ~isnns( varargin{ iEpsilon } )
                    msgStr = [ 'The regularization parameter must be a  ' ...
                               'non-negative scalar' ];
                    error( msgStr )
                end
                obj.epsilon = varargin{ iEpsilon };
            end
            if ~isempty( iRegType )
                if ~isrowstr( varargin{ iRegType } )
                    error( 'Invalid regularization type specification' )
                end
                obj.regType = varargin{ iRegType };
            end
        end
    end

end    
