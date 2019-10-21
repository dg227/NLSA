classdef nlsaDiffusionOperator_gl_mb_svd < nlsaDiffusionOperator_gl_mb
%NLSADIFFUSIONOPERATOR_GL_MB_SVD Class definition and constructor of diffusion 
% operator with multiple bandwidths and singular value decomposition instead
% of eigendecomposition. This class can be used to approximate PP^* and P^*P,
% where P is the integral operator approximated by diffusion maps.
% 
% Modified 2018/06/10   

    methods

        %% CLASS CONSTRUCTOR
        function obj = nlsaDiffusionOperator_gl_mb_svd( varargin )

            obj = obj@nlsaDiffusionOperator_gl_mb( varargin{ : } );

        end
    end

end    
