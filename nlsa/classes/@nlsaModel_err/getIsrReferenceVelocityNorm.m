function xiRef = getIsrReferenceVelocityNorm( obj, varargin )
% GETISRREFERENCEVELOCITYNORM Get ISR error reference velocity norm of an 
% nlsaModel_scl object
%
% Modified 2014/07/28

xiRef = getReferenceVelocityNorm( getIsrErrComponent( obj ), varargin{ : } );
