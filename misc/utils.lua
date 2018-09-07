local cjson = require 'cjson'

local utils = {}

function utils.vec2str( vector )
  content = ''
  if type(vector) == 'number' then vector = torch.FloatTensor({vector}) end
  vector:apply(function(x) 
                if not (x == 0) then 
                  content = content..itow[x]..' '
                end
              end)
  return content
end

return utils
