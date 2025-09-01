function xiRef = getOseReferenceVelocityNorm( obj, varargin )
% GETOSEREFERENCEVELOCITYNORM Get OSE error reference velocity norm of an 
% nlsaModel_scl object
%
% Modified 2014/07/28

xiRef = getReferenceVelocityNorm( getOseErrComponent( obj ), varargin{ : } );
