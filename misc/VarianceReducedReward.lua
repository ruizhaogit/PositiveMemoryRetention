local VarianceReducedReward, parent = torch.class("nn.VarianceReducedReward", "nn.Criterion")

function VarianceReducedReward:__init(module, scale)
   parent.__init(self)
   self.module = module
   self.scale = scale or 1
   self.sizeAverage = true
   self.gradInput = torch.Tensor()
end

function VarianceReducedReward:updateOutput(input, target)
   local input = self:toBatch(input, 1)
   self._reward = target
   self.reward = self.reward or input.new()
   self.reward:resize(self._reward:size(1)):copy(self._reward)
   self.reward:mul(self.scale)
   self.output = -self.reward:sum()
   if self.sizeAverage then
      self.output = self.output/input:size(1)
   end
   return self.output
end

function VarianceReducedReward:updateGradInput(input, target)
   self._reward = target:clone()
   self._reward:expandAs(self._reward, input)
   self.gradInput = self._reward:clone()
   return self.gradInput
end

function VarianceReducedReward:type(type)
   self._maxVal = nil
   self._maxIdx = nil
   self.__maxIdx = nil
   self._target = nil
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end
