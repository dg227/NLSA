function tag = getTag( obj )
% GETTAG Get tags of an nlsaComponent_rec_phi object
%
% Modified 2015/12/08

tag = getTag@nlsaComponent_rec( obj );
tag = { tag{ : } getBasisFunctionTag( obj ) };

