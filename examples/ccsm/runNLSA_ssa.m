function [ model, In ] = runNLSA_ssa( experiment )
%
% This function creates an SSA model and executes the various SSA steps for
%  data from the CCSM/CESM models 
%
% Each step saves the results on disk, so a partially completed calculation can
% be resumed by commenting out the steps in this function which have been 
% already exectuted. 
%
% Similarly, if the SSA model parameters specified in the function 
% ccsmNLSAModel_ssa are changed, it is only necessary to repeat the steps 
% affected by the parameter changes. 
%
% Input arguments:
%
% experiment:   a string identifier for the SSA model, passed to
%               the function ccsmNLSAModel_ssa
%
% ifPlot:       Set to true to make basic eigenfunction and density scatterplots
%               
% To recover the covariance eigenfunctions, and the projected and reconstructed
% data, run the following commands:
%
% Covariance eigenfunctions:
% phi = getCovarianceEigenfunctions( model ); 
% 
% Projected target data onto the diffusion eigenfunctions:
% a = getProjectedData( model );
%
% Reconstructed data: 
% x = getReconstructedData( model );
%  
% Modified 2016/06/02

% Default input arguments
if nargin == 0
    experiment = 'np_sst'; 
end

disp( experiment )
[ model, In ] = ccsmNLSAModel_ssa( experiment ); 


disp( 'Takens delay embedding' ); computeDelayEmbedding( model )

% The next step is only needed if the target data are different from the 
% source data
%disp( 'Takens delay embedding, target data' ); computeTrgDelayEmbedding( model )

fprintf( 'Covariance operator' ); 
computeCovarianceOperator( model )

disp( 'Projection of target data onto covariance eigenfunctions' );
computeProjection( model );

disp( 'Reconstruction of the projected data' )
computeReconstruction( model )
