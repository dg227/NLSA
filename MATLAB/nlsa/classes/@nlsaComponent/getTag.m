function tag = getTag( obj )
% GETTAG Get tags of an nlsaComponent object
%
% Modified 2014/07/28

tag = { getComponentTag( obj ) getRealizationTag( obj ) };
