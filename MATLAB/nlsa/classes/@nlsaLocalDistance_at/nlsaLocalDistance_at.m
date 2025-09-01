classdef nlsaLocalDistance_at < nlsaLocalDistance_l2
%NLSALOCALDISTANCE_AT  Class definition and constructor of local distance 
% scaled by the phase space velocity norm
%
% Modified 2014/04/01    

    %% PROPERTIES
    properties
        normalization = 'geometric';    % 'geometric' -> geometric mean
                                        % 'harmonic'  -> harmonic mean
    end

    methods
        %% NLSALOCALDISTANCE_AT  Class constructor
        function obj = nlsaLocalDistance_at( varargin )

            ifParentArg = true( 1, nargin );

            % Parse input arguments
            iNormalization = [];
            for i = 1 : 2 : nargin
                switch varargin{ i }
                    case 'normalization'
                        iNormalization = i + 1;
                        ifParentArg( [ i i + 1 ] ) = false;
                end
            end

            obj = obj@nlsaLocalDistance_l2( varargin{ ifParentArg } );
           
            % Set caller-defined values
            if ~isempty( iNormalization )
                if ischar( varargin{ iNormalization } )
                    if any( strcmp( varargin{ iNormalization }, { 'geometric' 'harmonic' } ) )
                        ifErr      = false;
                        obj.normalization = varargin{ iNormalization };
                    else
                        ifErr = true;
                    end
                else
                    ifErr = true;
                end
                if ifErr
                    error( 'Invalid normalization specification' )
                end
            end
        end
    end
end
