classdef nlsaIO

    properties 
        path = '.';
        file = '';
    end

    methods( Abstract )

        setDefaultPath( obj )

        setDefaultFile( obj )

    end
end
