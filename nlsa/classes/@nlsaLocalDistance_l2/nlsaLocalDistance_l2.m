classdef nlsaLocalDistance_l2 < nlsaLocalDistance
%NLSALOCALDISTANCE_L2  Class definition and constructor of local distance based 
% on L2 norm
%
% Modified 2014/04/07   

    methods

        %% NLSALOCALDISTANCE_L2  Class constructor
        function obj = nlsaLocalDistance_l2( varargin )
            obj = obj@nlsaLocalDistance( varargin{ : } );
        end
                        
    end
end    
