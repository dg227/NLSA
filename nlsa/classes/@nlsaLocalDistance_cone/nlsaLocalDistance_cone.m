classdef nlsaLocalDistance_cone < nlsaLocalDistance_at
%NLSALOCALDISTANCE_CONE  Class definition and constructor of cone 
% distance 
%
% Modified 2015/11/19 

    %% PROPERTIES
    properties
        alpha = 1;
        zeta  = 0.99;
        tol   = 0;
    end

    methods

        %% NLSALOCALDISTANCE_CONE  Class constructor
        function obj = nlsaLocalDistance_cone( varargin )
            ifParentArg    = true( 1, nargin );
            
            % Parse input arguments
            iAlpha   = [];
            iZeta    = [];
            iTol     = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'alpha'
                        iAlpha = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false; 
                    case 'zeta'
                        iZeta = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                    case 'tolerance'
                        iTol = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false; 
                end
            end

            obj = obj@nlsaLocalDistance_at( varargin{ ifParentArg } );

            % Set caller-defined values
            if ~isempty( iAlpha )
                if ~isrs( varargin{ iAlpha } ) 
                    error( 'Alpha must be a real scalar' )
                end
                obj.alpha = varargin{ iAlpha };
            end
            if ~isempty( iZeta )
                if ~isscalar( varargin{ iZeta } ) ...
                  || ~( varargin{ iZeta } < 1 )
                    error( 'zeta must be less than 1' )
                end
                obj.zeta = varargin{ iZeta };
            end
            if ~isempty( iTol )
                if ~isscalar( varargin{ iTol } ) ...
                  || ~( varargin{ iTol } >= 0 )
                    error( 'Tolerance must be a non-negative scalar' )
                end
                obj.tol = varargin{ iTol };
            end
        end
    end
end
