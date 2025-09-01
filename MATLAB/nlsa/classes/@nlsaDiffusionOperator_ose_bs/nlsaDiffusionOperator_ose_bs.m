classdef nlsaDiffusionOperator_ose_bs < nlsaDiffusionOperator_ose_svd
%NLSADIFFUSIONOPERATOR_OSE_BS Class definition and constructor of diffusion 
%  operator with bistochastic kernel and out of sample extension (OSE),
% 
% Modified 2018/06/15
    methods

        function obj = nlsaDiffusionOperator_ose_bs( varargin )

            obj = obj@nlsaDiffusionOperator_ose_svd( varargin{ :  } );
        end
    end
end    
