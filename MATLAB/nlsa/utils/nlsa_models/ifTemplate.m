function ifTempl = ifTemplate( className, varargin )                
%  IFTEMPLATE  Helper function to check whether nlsaModel constructors are called 
%  in template or direct mode
%
%  Modified 2014/02/06

eval( [ 'templateName = ' className '.listParserProperties;' ] )   
ifTempl = false;
for i = 1 : 2 : nargin - 1
    if any( strcmp( varargin{ i }, templateName ) )
        ifTempl = true;
        break
    end
end 
