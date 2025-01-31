function s = demoKoopman_experiment( P )
% DEMOKOOPMAN_EXPERIMENT Construct string identifier for analysis experiment.


s = strjoin_e( { P.dataset ...
                 strjoin_e( P.period, '-' ) ... 
                 strjoin_e( P.sourceVar, '_' ) ...
                 sprintf( 'emb%i', P.embWindow ) ...
                 P.kernel }, ...
               '_' );

if isfield( P, 'ifDen' )
    if P.ifDen
        s = [ s '_den' ];
    end
end
