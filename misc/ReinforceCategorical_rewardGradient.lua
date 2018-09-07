local ReinforceCategorical_rewardGradient, parent = torch.class("nn.ReinforceCategorical_rewardGradient", "nn.Reinforce_rewardGradient")

function ReinforceCategorical_rewardGradient:updateOutput(input)
   self.stochastic = self.stochastic or false
   self.output:resizeAs(input)
   self._index = self._index or ((torch.type(input) == 'torch.CudaTensor') and torch.CudaLongTensor() or torch.LongTensor())-- CudaTensor()
   if self.stochastic then
      self._input = self._input or input.new()
      self._input:resizeAs(input):copy(input):add(0.00000001) 
      input.multinomial(self._index, self._input, 1) -- 1
      self.output:zero()
      self.output:scatter(2, self._index, 1)
   else
      self.output:copy(input)
   end
   return self.output
end

function ReinforceCategorical_rewardGradient:updateGradInput(input, gradOutput)
   -- Note that gradOutput is ignored
   -- f : categorical probability mass function
   -- x : the sampled indices (one per sample) (self.output)
   -- p : probability vector (p[1], p[2], ..., p[k])  
   -- derivative of log categorical w.r.t. p
   -- d ln(f(x,p))     1/p[i]    if i = x  
   -- ------------ =   
   --     d p          0         otherwise
   self.gradInput = gradOutput:clone()
   self._input = self._input or input.new()
   self._input:resizeAs(input):copy(input):add(0.00000001) 
   self.gradInput:cdiv(self._input)
   self.gradInput:mul(-1)
   return self.gradInput
end

function ReinforceCategorical_rewardGradient:type(type, tc)
   self._index = nil
   return parent.type(self, type, tc)
end
