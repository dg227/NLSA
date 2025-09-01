function y = bump(x)
    y = zeros(size(x));
    ifX = x > -1 & x < 1;
    y(ifX) = exp(-1 ./ (1 - x(ifX)));
end
