function dSum = getDoubleSum( obj )
% GETDOUBLESUM  Read double sum of an nlsaDiffusionOperator_gl_fb object
%
% Modified 2015/05/08

load( fullfile( getOperatorPath( obj ), getDoubleSumFile( obj ) ), ...
      'dSum' )
