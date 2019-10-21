classdef nlsaLocalScaling_pwr < nlsaLocalScaling
%NLSALOCALSCALING_PWR  Class definition and constructor of local distance 
% scaling by a power law

%
% Modified 2015/10/23

    %% PROPERTIES
    properties
        p    = 1; % exponent
    end

    methods
        %% NLSALOCALSCALING_PWR  Class constructor
        function obj = nlsaLocalScaling_pwr( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iP   = [];

            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'pwr'
                        iP = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalScaling( varargin{ ifParentArg } );
           
            % Set caller-defined values
            if ~isempty( iP )
                if ~isscalar( varargin{ iP } )
                    error( 'Exponent parameter must be a scalar' )
                end
                obj.p = varargin{ iP };
            end
        end
    end
end
