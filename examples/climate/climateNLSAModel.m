function [ model, In, Out ] = climateNLSAModel( dataset, experiment )
%
% CLIMATENLSAMODEL Construct NLSA model for climate data.
%
% Input arguments:
%
% experiment:   a string identifier for the data analysis experiment
%
% dataset:      a string identifier for the dataset. Possible options are:
%
%               - noaa    (NOAA 20th century reanalysis)
%               - hadisst (HadISST dataset) 
%               - ccsm4   (CCSM4 model)
%               - claus   (CLAUS brightness temperature dataset)
%               - gpcp    (GPCP griddded precipitation dataset)
% 
% climateNLSAModel calls dataset-speficic model constructors according to the
% dataset argument. 
%
% In and Out are data structures containing the model parameters. 
%
% Modified 2020/03/27

% Default input arguments
if nargin == 0 
    dataset = 'noaa';
end
if nargin <= 1
    experiment = 'enso_lifecycle';   
end

switch dataset
case 'noaa'
    [ model, In, Out ] = noaaNLSAModel( experiment ); 
case 'hadisst'
    [ model, In, Out ] = hadisstNLSAModel( experiment ); 
case 'ccsm4'
    [ model, In, Out ] = ccsm4NLSAModel( experiment ); 
case 'gpcp'
    [ model, In, Out ] = gpcpNLSAModel( experiment ); 
case 'claus'
    [ model, In, Out ] = clausNLSAModel( experiment ); 
otherwise
    error( 'Invalid dataset' )
end



