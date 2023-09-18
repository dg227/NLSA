function b = isleapyear(yr)
    % Determine if one or more years are leap years.
    %
    % To determine whether a year is a leap year, we follow these steps:
    % If the year is evenly divisible by 4, go to step 2. Otherwise, go to step 5.
    % If the year is evenly divisible by 100, go to step 3. Otherwise, go to step 4.
    % If the year is evenly divisible by 400, go to step 4. Otherwise, go to step 5.
    % The year is a leap year (it has 366 days).
    % The year is not a leap year (it has 365 days).
    %
    % Source: https://learn.microsoft.com/en-us/office/troubleshoot/excel/determine-a-leap-year
    %
    % Modified 2023/09/06
    b = mod(yr, 4) == 0 ...
        && (mod(yr, 100) ~= 0 || (mod(yr, 100) == 0 && mod(yr, 400) == 0));
end
