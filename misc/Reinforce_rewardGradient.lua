local Reinforce_rewardGradient, parent = torch.class("nn.Reinforce_rewardGradient", "nn.Module")

function Reinforce_rewardGradient:__init(stochastic)
   parent.__init(self)
   self.stochastic = stochastic
end

function Reinforce_rewardGradient:updateOutput(input)
   self.output:set(input)
end

function Reinforce_rewardGradient:updateGradInput(input, gradOutput)
   local reward = self:rewardAs(input)
   self.gradInput:resizeAs(reward):copy(reward)
end

function Reinforce_rewardGradient:rewardAs(input)
   if self.reward == nil then 
      self.reward = torch.Tensor(input:size()):fill(0):sum(2):squeeze()
   end
   assert(self.reward:dim() == 1)
   if input:isSameSizeAs(self.reward) then
      return self.reward
   else
      if self.reward:size(1) ~= input:size(1) then
         input = input:sub(1,self.reward:size(1))
         assert(self.reward:size(1) == input:size(1), self.reward:size(1).." ~= "..input:size(1))
      end
      self._reward = self._reward or self.reward.new()
      self.__reward = self.__reward or self.reward.new()
      local size = input:size():fill(1):totable()
      size[1] = self.reward:size(1)
      self._reward:view(self.reward, table.unpack(size))
      self.__reward:expandAs(self._reward, input)
      return self.__reward
   end
end
