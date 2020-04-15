function df = computeFD( obj, f )
% COMPUTEFD Compute finite difference of time-ordered data.
%
% This function operates along the second (column) dimension of array f.
%
% Modified 2020/04/15

% Validate input arguments
if ~isnumeric( f ) || ~ismatrix( f )
    error( 'Second input argument must be a numeric matrix' )
end

% Determine basic array sizes, finite difference weights
nD   = size( f, 1 );          % data dimension
nS   = size( f, 2 );          % number of input samples
w    = getFDWeights( obj );   % finite-differene weights
nW   = numel( w );            % number of weights
nSFD = nS - nW + 1;           % number of output samples

% Initialize output array, and perform FD
df = zeros( nD, nSFD );
for iW = 1 : nW
    df = df + w( iW ) * f( :, iW : iW + nSFD - 1 );
end
