classdef nlsaKernelDensity_ose < nlsaKernelDensity
% NLSAKERNELDENSITY_OSE Class definition and constructor for kernel density
%  estimation for out-of-sample data
% 
% Modified 2018/07/06
    methods

    % Bandwidth parameter epsilon for this class is a multiplier of 
    % the bandwidth for the in-sample data

        function obj = nlsaKernelDensity_ose( varargin )

            obj = obj@nlsaKernelDensity( varargin{ :  } );
        end
    end
end    
