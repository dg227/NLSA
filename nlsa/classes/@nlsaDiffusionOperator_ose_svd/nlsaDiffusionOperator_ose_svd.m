classdef nlsaDiffusionOperator_ose_svd < nlsaDiffusionOperator_ose
%NLSADIFFUSIONOPERATOR_OSE Class definition and constructor of diffusion 
%  operator with out of sample extension (OSE) using SVD.
% 
% Modified 2018/06/13
    methods

        function obj = nlsaDiffusionOperator_ose_svd( varargin )

            obj = obj@nlsaDiffusionOperator_ose( varargin{ :  } );
        end
    end
end    
