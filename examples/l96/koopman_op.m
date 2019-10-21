function [ A, B, D, q, E ] = koopman_op( phi, s, Aeta, q1, q2, nLA, nLX )

sHat = fft( s, [], 1 );

nZ = nLA * nLX ^ 2; 

nnZ = 2 * nLA ^ 2 * nLX ^ 2 + nLA ^ 2 * nLX ^ 3;


iA = zeros( nNZ, 1 );
jA = zeros( nNZ, 1 );
iB = zeros( nNZ, 1 );
jB = zeros( nNZ, 1 );
vA = zeros( nNZ, 1 );
vB = zeros( nNZ, 1 );
vD = zeros( nNZ, 1 );
E  = zeros( nZ, 1 );
q  = zeros( nZ, 4 );

kA = 1;
kB = 1;
jK = 1;
for jX2 = 1 : nLX
    for jX1 = 1 : nLX
        for jA1 = 1 : nLA
            iK = 1;
            for iX2 = 1 : nLX
                for iX1 = 1 : nLX
                    for iA1 = 1 : nLA
                        if     all( [ jA1 jA2 ] == - nLA ) ...
                            && all( [ jX1 jX2 ] == - nLX ) 
                            E( iK ) =  eta( iA ) + iA2 .^ 2 ...
                                      + iX1 .^ 2 + iX2 .^ 2;
                                    q( iK, : ) = [ iA1 iA2 iX1 iX2 ];
                                end
                                ifSetA = false;
                                ifSetB = false;
                                if iA1 == jA1 && ...
                                   iA2 == jA2 && ...
                                   iX1 == jX1 && ...
                                   iX2 == jX2 
                                    ifSetA = true;
                                    ifSetB = true;
                                    vA( kA ) = omega1 * jA1 + omega2 * jA2;
                                    vB( kB ) = 1 / max( 1, E( iK ) );
                                    vD( kB ) = sign( E( iK ) );
                                end
                                if    iA1 == jA1 - 1 ...
                                   && jA2 == iA2 ...
                                   && iX1 == jX1 ...
                                   && iX2 == jX2 + q2
                                    ifSetA = true;
                                    vA( kA ) = .5 * V1 * jX1;
                                end
                                if    iA1 == jA1 + 1 ...
                                   && jA2 == iA2 ...
                                   && iX1 == jX1 ...
                                   && iX2 == jX2 - q2
                                    ifSetA = true;
                                    vA( kA ) = .5 * V1 * jX1;
                                end
                                if    iA1 == jA1 ...
                                   && iA2 == jA2 - 1 ...
                                   && iX1 == jX1 + q1 ...
                                   && iX2 == jX2
                                    ifSetA = true;
                                    vA( kA ) = .5 * V2 * jX2;
                                end
                                if    iA1 == jA1 ...
                                   && iA2 == jA2 + 1 ...
                                   && iX1 == jX1 - q1 ...
                                   && iX2 == jX2
                                    ifSetA = true;
                                    vA( kA ) = .5 * V2 * jX2;
                                end 
                                if ifSetA 
                                    iA( kA ) = iK;
                                    jA( kA ) = jK;
                                    kA = kA + 1;
                                end
                                if ifSetB
                                    iB( kB ) = iK;
                                    jB( kB ) = jK;
                                    kB = kB + 1;
                                end          
                                iK = iK + 1;
                            end % iA1
                        end % iA2
                    end % iX1
                end % iX2
                jK = jK + 1;
            end % jA1
        end % jA2
    end % jX1
end % jX2
                                
iA = iA( 1 : kA - 1 );
jA = jA( 1 : kA - 1 );
iB = iB( 1 : kB - 1 );
jB = jB( 1 : kB - 1 );
vA = vA( 1 : kA - 1 );
vB = vB( 1 : kB - 1 );
vD = vD( 1 : kB - 1 );
A = sparse( iA, jA, i * vA, nZ, nZ );                                         
B = sparse( iB, jB, vB, nZ, nZ );
D = sparse( iB, jB, vD, nZ , nZ );

