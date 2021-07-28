function [ yExp, yStd, yPrb ] = qmda( K, U, A, A2, AEvec, specIdx, xi0, nTO, ...
                                      nPar )

ifPrb = nargout == 3;

nDA = size( K, 3 );
nTF = size( U, 3 );
nY  = size( A, 3 );

if nargin < 9
    nPar = 0;
end

yExp = zeros( nY, nTF + 1, nDA + 1 );
yStd = zeros( nY, nTF + 1, nDA + 1 );
if ifPrb 
    nQ = size( specIdx, 1 );
    yPrb = zeros( nQ, nY, nTF + 1, nDA + 1 );
end

%% DA INITIALIZATION

disp( 'DA initialization...' ); 
t = tic; 

% Compute expecation, standard deviation, and (if requested) probability 
% distribution of the target observables based on the initial state. 
[ yExp( :, 1, 1 ), yStd( :, 1, 1 ) ] =  qmExpStd( A, A2, xi0 );
if ifPrb
    yPrb( :, :, 1, 1 ) = qmPrb( AEvec, specIdx, xi0 );
end


% Initial forecast step: Evolve wavefunction by Koopman operators and compute 
% the corresponding expectation and standard deviation of the target 
% observables. We also return the wavefunction at the observational 
% timeshift nTO as the prior wavefunction, xiPrior, to be used in the  main
% assimilation cycle.
if ifPrb
    [ xiPrior, ...
      yExp( :, 2 : nTF + 1, 1 ), yStd( :, 2 : nTF + 1, 1 ), ...
      yPrb( :, :, 2 : nTF + 1, 1 ) ] = ...
        qmdaForecast( xi0, U, A, A2, AEvec, specIdx, nTO, nPar );  
else
    [ xiPrior, ...
      yExp( :, 2 : nTF + 1, 1 ), ...
      yStd( :, 2 : nTF + 1, 1 ) ] = ...
        qmdaForecast( xi0, U, A, A2, [], [], nTO, nPar );  
end
toc( t )

%% DA MAIN LOOP
disp( 'QMDA main loop:' )
nCount = 10;
for iDA = 1 : nDA  
    if mod( iDA - 1, nCount ) == 0
        disp( [ 'Step ' int2str( iDA ) '/' int2str( nDA ) '...' ] ); 
        t = tic;
        iCount = 0;
    end

    % Analysis step: Update wavefunction based on feature operator at the 
    % current assimilation step.
    xiPost = qmdaAnalysis( xiPrior, K( :, :, iDA ) ); 

    % Compute the expecation, standard deviation, and (if requested) 
    % probability distribution of the target observables after the analysis 
    % step. 
    [ yExp( :, 1, iDA + 1 ), yStd( :, 1, iDA + 1 ) ] ...
        = qmExpStd( A, A2, xiPost );
    if ifPrb
        yPrb( :, :, 1, iDA + 1 ) = qmPrb( AEvec, specIdx, xiPost ); 
    end

    % Forecast step: Evolve wavefunction by Koopman operators and compute the 
    % corresponding expectation, standard deviation, and probability 
    % of the target observables. We also return the wavefunction at the 
    % observational timeshift nTO as the prior wavefunction, xiPrior, to be 
    % used in the assimilation cycle.
    if ifPrb
        [ xiPrior, ...
          yExp( :, 2 : nTF + 1, iDA + 1 ), ...
          yStd( :, 2 : nTF + 1, iDA + 1 ), ...
          yPrb( :, :, 2 : nTF + 1, iDA + 1 ) ] = ...
            qmdaForecast( xiPost, U, A, A2, AEvec, specIdx, nTO, nPar );  
    else
        [ xiPrior, ...
          yExp( :, 2 : nTF + 1, iDA + 1 ), ...
          yStd( :, 2 : nTF + 1, iDA + 1 ) ] = ...
            qmdaForecast( xiPost, U, A, A2, [], [], nTO, nPar );  
    end

    iCount = iCount + 1;
    if iCount == nCount
        toc( t )
    end
end

end


%% AUXILIARY FUNCTIONS
function [ yE, yS ] = qmExpStd( A, A2, xi )
yE = qmExpectation( A, xi );
yS = sqrt( abs( qmExpectation( A2, xi ) - yE .^ 2 ) );
end

function [ xi1, yE, yS, yP ] = qmdaForecast( xi0, U, A, A2, AEvec, specIdx, ...
                                             nTO, nPar );
ifPrb = nargout == 4;
nT  = size( U, 3 );
nY  = size( A, 3 );
yE  = zeros( nY, nT );
yS  = zeros( nY, nT );
if ifPrb
    nQ = size( specIdx, 1 );
    yP = zeros( nQ, nY, nT );
end
xi1 = xi0; % Initially assigned here because of parallel for loop  
parfor( iT = 1 : nT, nPar )
   xi = xi0 * U( :, :, iT );
   xi = xi / norm( xi );
   if iT == nTO
       xi1 = xi;
   end
   [ yE( :, iT ), yS( :, iT ) ] = qmExpStd( A, A2, xi );
   if ifPrb 
       yP( :, :, iT ) = qmPrb( AEvec, specIdx, xi );
   end
end
end

function xi1 = qmdaAnalysis( xi0, K )
xi1 = xi0 * K;
xi1 = xi1 / norm( xi1 );
end

function p = qmPrb( AEvec, specIdx, xi );
nY = size( AEvec, 3 );
nQ = size( specIdx, 1 );
p  = zeros( nQ, nY );
for iY = 1 : nY
    for iQ = 1 : nQ
        idxQ = specIdx( iQ, 1, iY ) : specIdx( iQ, 2, iY );
        p( iQ, iY ) = sum( abs( xi * AEvec( :, idxQ, iY  ) ) .^ 2 ); 
    end
end
end
