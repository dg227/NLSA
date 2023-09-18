function y = pos_freq(x)
    % Orthogonal projection onto positive frequency subspace.
    %
    % Modified 2023/07/21

    n = size(x, 1);
    n_pos = floor((n - 1) / 2);
    x_hat = fft(x, n, 1);
    x_hat(1, :) = 0;
    x_hat(2 + n_pos : end, :) = 0;
    y = ifft(x_hat, n, 1);
end
