classdef nlsaLocalDistance_sone < nlsaLocalDistance_at
%NLSALOCALDISTANCE_SONE  Class definition and constructor of sine cone 
% distance 
%
% Modified 2015/03/30 

    %% PROPERTIES
    properties
        ifVNorm = true;
        zeta    = 0.99;
        tol     = 0;
    end

    methods

        %% NLSALOCALDISTANCE_SONE  Class constructor
        function obj = nlsaLocalDistance_sone( varargin )
            ifParentArg    = true( 1, nargin );
            
            % Parse input arguments
            iIfVNorm = [];
            iZeta    = [];
            iTol     = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'ifVNorm'
                        iIfVNorm = i + 1;
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
            if ~isempty( iIfVNorm )
               if ~isscalar( varargin{ iIfVNorm } ) ...
                 || ~islogical( varargin{ iIfVNorm } )
                   error( 'ifVNorm must be a logical scalar' )
               end
               obj.ifVNorm = varargin{ iIfVNorm };
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
