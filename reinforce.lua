require 'optim'
require 'rnn'
require 'hdf5'
require 'xlua'
require 'nngraph'
cjson = require 'cjson'
utils = require 'misc.utils'
dofile("misc/optim_updates.lua")
dofile("misc/vocab.lua")
dofile("misc/VarianceReducedReward.lua")
dofile("misc/Reinforce_rewardGradient.lua")
dofile("misc/ReinforceCategorical_rewardGradient.lua")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(123)

cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('--use_cuda', true, 'use_cuda')
cmd:option('--batchSize', 256, 'batchSize')
cmd:option('--epochs', 0, 'epochs')
cmd:option('--retention', false, 'use Positive Memory Retention')
cmd:option('--updateBuffer', true, 'use Probability Updating')
cmd:option('--trustFactor', 10, 'the value of the Importance Weight Upper Bound')
cmd:option('--qgen_model', 'qgen_float.t7', 'load the pretrained qgen_model')
cmd:text()
local opt = cmd:parse(arg or {})

local use_cuda = opt.use_cuda
local debug = false

local quesLen = 7
local batchSize = opt.batchSize
local epochs = opt.epochs
local deviceId = 1
local rounds = 4

local retention = opt.retention
local remeberThresh = 0
local updateBuffer = opt.updateBuffer
local trustFactor = opt.trustFactor
local trustFactor_inverse = 1/trustFactor
local earlyStopThresh = 2
local repTresh = 10
local use_save_copy = true

local use_baseline1 = true
local use_baseline2 = false

local state = {}
state.learningRate = 0.0001
local load_model = true
running_average = 0
running_average_z = 0
local savepath_retention

local qgen_model = opt.qgen_model


print('quesLen', quesLen, 'batchSize', batchSize, 'use_cuda', use_cuda, 'deviceId', deviceId ,'rounds', rounds, 
  'retention', retention, 'debug', debug, 'updateBuffer', updateBuffer, 'trustFactor', trustFactor, 
  'earlyStopThresh', earlyStopThresh, 'use_baseline1', use_baseline1, 'use_baseline2', use_baseline2, 
  'learningRate', state.learningRate, 'remeberThresh', remeberThresh, 'repTresh', repTresh, 
  'use_save_copy', use_save_copy, 'trustFactor_inverse', trustFactor_inverse, 'epochs', epochs
   )

if use_cuda then 
  require 'cunn' 
  require 'cutorch'
  cutorch.setDevice(deviceId)
  cutorch.setHeapTracking(true)
  cutorch.manualSeed(123)
end

model_path = 'rf_qgen '..os.date("%c") .. '.t7'

fd = io.open('sample.txt', 'a')
fd:write('\n'..'******************* '..os.date("%c")..' *******************'..'\n')
fd:close()

function load_data( split )
  local Size = 0 
  for line in io.lines('data/' .. split .. '.json') do 
    local example = cjson.decode(line)
    Size = Size + 1 
    gridSize = #example['gridImg'] 
  end

  local image_bgcolors = torch.FloatTensor(Size, gridSize^2):fill(0)
  local image_styles = torch.FloatTensor(Size, gridSize^2):fill(0)
  local image_colors = torch.FloatTensor(Size, gridSize^2):fill(0)
  local image_numbers = torch.FloatTensor(Size, gridSize^2):fill(0)

  local target_bgcolors = torch.FloatTensor(Size):fill(0)
  local target_styles = torch.FloatTensor(Size):fill(0)
  local target_colors = torch.FloatTensor(Size):fill(0)
  local target_numbers = torch.FloatTensor(Size):fill(0)

  local labels = torch.FloatTensor(Size):fill(0)

  local i = 1
  for line in io.lines('data/' .. split .. '.json') do
    local example = cjson.decode(line)
    for x = 1, gridSize do 
      for y = 1, gridSize do
        local cell = example['gridImg'][x][y]
        image_bgcolors[i][(x-1)*gridSize+y] = bgcolorstoi[cell['bgcolor']]
        image_styles[i][(x-1)*gridSize+y] = stylestoi[cell['style']]
        image_colors[i][(x-1)*gridSize+y] = colorstoi[cell['color']]
        image_numbers[i][(x-1)*gridSize+y] = cell['number']+1
      end
    end

    if split == 'train' then
      example['target'][1], example['target'][2] = torch.randperm(3)[1]-1, torch.randperm(3)[1]-1
    end

    labels[i] = example['target'][1]*gridSize+example['target'][2]+1
    local cell = example['gridImg'][example['target'][1]+1][example['target'][2]+1]
    target_bgcolors[i] = bgcolorstoi[cell['bgcolor']]
    target_styles[i] = stylestoi[cell['style']]
    target_colors[i] = colorstoi[cell['color']]
    target_numbers[i] = cell['number']+1

    i = i + 1
  end
  return Size, image_bgcolors, image_styles, image_colors, image_numbers, labels, target_bgcolors, target_styles, target_colors, target_numbers
end

trainSize, train_image_bgcolors, train_image_styles, train_image_colors, train_image_numbers, train_labels, 
train_target_bgcolors, train_target_styles, train_target_colors, train_target_numbers = load_data('train')

validSize, valid_image_bgcolors, valid_image_styles, valid_image_colors, valid_image_numbers, valid_labels, 
valid_target_bgcolors, valid_target_styles, valid_target_colors, valid_target_numbers = load_data('valid')

testSize, test_image_bgcolors, test_image_styles, test_image_colors, test_image_numbers, test_labels, 
test_target_bgcolors, test_target_styles, test_target_colors, test_target_numbers = load_data('test')

shuffle = torch.randperm(trainSize)

dofile("misc/qgen.lua")

softmax =
  rnn_cell
- nn.Linear(hiddenSize, #itow)
- nn.SoftMax()

output =
  softmax
- nn.ReinforceCategorical_rewardGradient()

output:annotate{name = 'output'}

table.insert(outputs, output)

qgen = nn.gModule(inputs, outputs)
qgen = nn.Recursor(qgen, (quesLen+1)*rounds)

oracle = torch.load('oracle_float.t7')
guesser = torch.load('guesser_float.t7')

qgenW, qgendW = qgen:getParameters()
qgenW:copy( torch.load(qgen_model):getParameters() )

criterion_qgen = nn.VarianceReducedReward(qgen)

function sampling(qgen_input, oracle_input, seqLen, split)
  qgen:evaluate()
  qgen:forget()
  local word_i = torch.FloatTensor(batchSize):fill(wtoi['<START>'])
  local outputs = {}
  local ques = torch.FloatTensor(batchSize, quesLen):fill(0)
  local qas = torch.FloatTensor(batchSize, (quesLen+1)*rounds):fill(0)
  local prob, nextToken

  if use_cuda then ques:cuda() qas:cuda() end

  for i=1, rounds*(seqLen+1) do
    if use_cuda then word_i:cuda() nn.utils.recursiveType(qgen_input, 'torch.CudaTensor') end
    outputs[i] = qgen:forward({word_i, unpack(qgen_input)})
    if type(outputs[i]) == 'table' then
      outputs[i] = outputs[i][1]
    end
    if not (split == 'train') then
      prob, nextToken = torch.max(outputs[i], 2)
    else
      nextToken = torch.multinomial(outputs[i], 1)
    end
    if not (i%(quesLen+1) == 0) then
      word_i = nextToken:float():squeeze()
      if use_cuda then word_i:cuda() end
      ques[{{},{i%(quesLen+1)}}] = word_i:clone()
      qas[{{},{i}}] = word_i:clone()
    else
      if use_cuda then ques = ques:cuda() nn.utils.recursiveType(oracle_input, 'torch.CudaTensor') end
      local ans = oracle:forward({ques, unpack(oracle_input)})
      local prob, ans = torch.max(torch.exp(ans), 2)
      ans = ans:float():squeeze()
      if use_cuda then ans:cuda() end
      for i=1, batchSize do
        ans[i] = wtoi[itoanswer[ans[i]]]
      end
      word_i = ans
      qas[{{},{i}}] = word_i:clone()
      ques = ques:fill(0)
    end

    local word = word_i:float()[1] 
    fd = io.open('sample.txt', 'a')
    fd:write(utils.vec2str(word))
    fd:close()
  end
  fd = io.open('sample.txt', 'a')
  fd:write('\n')
  fd:close()

  return qas
end

function validating( split )
  running_average = 0
  for t = 1, torch.floor(_G[split..'Size']/batchSize)*batchSize, batchSize do
    xlua.progress(torch.floor(t/batchSize)+1, torch.floor(_G[split..'Size']/batchSize))
    qgen:evaluate()
    oracle:evaluate()
    guesser:evaluate()
    qgen:forget()

    if use_cuda then
      qgen:cuda()
      oracle:cuda()
      guesser:cuda()
      criterion_qgen:cuda()
    end

    local qgen_input = { _G[split..'_image_bgcolors'][{{t, t+batchSize-1}}], _G[split..'_image_styles'][{{t, t+batchSize-1}}], 
    _G[split..'_image_colors'][{{t, t+batchSize-1}}], _G[split..'_image_numbers'][{{t, t+batchSize-1}}] }
    local oracle_input = { _G[split..'_target_bgcolors'][{{t, t+batchSize-1}}], _G[split..'_target_styles'][{{t, t+batchSize-1}}], 
    _G[split..'_target_colors'][{{t, t+batchSize-1}}], _G[split..'_target_numbers'][{{t, t+batchSize-1}}] }
    local qas = sampling(qgen_input, oracle_input, quesLen, split)
    local guesser_input = { qas, unpack(qgen_input) }
    local guesser_label = _G[split..'_labels'][{{t, t+batchSize-1}}]
    if use_cuda then nn.utils.recursiveType(guesser_input, 'torch.CudaTensor') end
    local guesser_output = guesser:forward(guesser_input)
    local prob, predicts = torch.max(guesser_output, guesser_output:dim())
    if use_cuda then 
      guesser_label = guesser_label:type("torch.CudaLongTensor")
    else
      guesser_label = guesser_label:long()
    end
    local reward = torch.eq(predicts, guesser_label):float()
    running_average = ( (t-1)*running_average + (batchSize)*(reward:mean()) ) / (t-1+batchSize)
  end
end

if retention then
  trajectory_qas = torch.FloatTensor(trainSize, (quesLen+1)*rounds):fill(0)
  trajectory_image_bgcolors = torch.FloatTensor(trainSize, gridSize^2):fill(0)
  trajectory_image_styles = torch.FloatTensor(trainSize, gridSize^2):fill(0)
  trajectory_image_colors = torch.FloatTensor(trainSize, gridSize^2):fill(0)
  trajectory_image_numbers = torch.FloatTensor(trainSize, gridSize^2):fill(0)
  trajectory_reward = torch.FloatTensor(trainSize):fill(0)
  trajectory_baseline = torch.FloatTensor(trainSize):fill(0)
  trajectory_prob = torch.FloatTensor(trainSize, (quesLen+1)*rounds):fill(0)
  pointer = 0
end

if debug then batchSize = 3 trainSize = 12*batchSize validSize = 3*batchSize end

best_valid = 0
for e = 1, epochs do
  running_average = 0
  if savepath_retention and use_save_copy then 
    qgenW:copy( torch.load(savepath_retention):getParameters() )
    print('Loading retention model ...')
  end
  for t = 1, torch.floor(trainSize/batchSize)*batchSize, batchSize do
    xlua.progress(torch.floor(t/batchSize)+1, torch.floor(trainSize/batchSize))
    qgen:evaluate()
    oracle:evaluate()
    guesser:evaluate()
    qgen:forget()

    if use_cuda then
      qgen:cuda()
      oracle:cuda()
      guesser:cuda()
      criterion_qgen:cuda()
    end

    local qgen_input = { train_image_bgcolors[{{t, t+batchSize-1}}], train_image_styles[{{t, t+batchSize-1}}], 
    train_image_colors[{{t, t+batchSize-1}}], train_image_numbers[{{t, t+batchSize-1}}] }
    local oracle_input = { train_target_bgcolors[{{t, t+batchSize-1}}], train_target_styles[{{t, t+batchSize-1}}], 
    train_target_colors[{{t, t+batchSize-1}}], train_target_numbers[{{t, t+batchSize-1}}] }
    local qas = sampling(qgen_input, oracle_input, quesLen, 'train')
    local guesser_input = { qas, unpack(qgen_input) }
    local guesser_label = train_labels[{{t, t+batchSize-1}}]
    if use_cuda then nn.utils.recursiveType(guesser_input, 'torch.CudaTensor') end
    local guesser_output = guesser:forward(guesser_input)
    local prob, predicts = torch.max(guesser_output, guesser_output:dim())
    if use_cuda then 
      guesser_label = guesser_label:type("torch.CudaLongTensor")
    else
      guesser_label = guesser_label:long()
    end
    local reward = torch.eq(predicts, guesser_label):float()
    running_average = ( (t-1)*running_average + (batchSize)*(reward:mean()) ) / (t-1+batchSize)

    local qas_i = torch.FloatTensor(batchSize, rounds*(quesLen+1)):fill(0)
    local qas_o = torch.FloatTensor(batchSize, rounds*(quesLen+1)):fill(0)
    qas_i[{{},{1}}]:fill(wtoi['<START>'])
    qas_i[{{},{2,-1}}] = qas[{{},{1,-2}}]:clone()
    qas_o[{{},{1,-1}}] = qas:clone()

    qgen:training()
    qgen:zeroGradParameters()
    qgen:forget()

    local reward_target
    local outputs, reward_targets = {}, {}
    local seqLen = rounds*(quesLen+1)
    for i=1,seqLen do 
      local word_i = qas_i[{{},{i}}]:squeeze()
      local word_o = qas_o[{{},{i}}]:squeeze()
      if use_cuda then word_i:cuda() nn.utils.recursiveType(qgen_input, 'torch.CudaTensor') end
      outputs[i] = qgen:forward({word_i, unpack(qgen_input)})
      reward_target = outputs[i]:sum(2):squeeze():clone():fill(0)
      local baseline_target = reward_target:clone()

      for j=1, batchSize do
          if reward:squeeze()[j] == 1 then
            reward_target[j] = 1 
          end
          baseline_target[j] = running_average 
      end
      if use_baseline1 then 
        reward_target = reward_target - baseline_target
      end
      reward_targets[i] = reward_target:view(reward_target:size(1), 1):clone()
      if use_cuda then outputs[i]:cuda() reward_targets[i]:cuda() end
      criterion_qgen:forward(outputs[i], reward_targets[i])
    end

    local gradOutputs, gradInputs = {}, {}
    for i=seqLen,1,-1 do 
      local word_i = qas_i[{{},{i}}]:squeeze()
      local word_o = qas_o[{{},{i}}]:squeeze()
      if use_cuda then outputs[i]:cuda() reward_targets[i]:cuda() end
      gradOutputs[i]  = criterion_qgen:backward(outputs[i], reward_targets[i])
      local reward_mask = torch.FloatTensor(gradOutputs[i]:size()):fill(0)
      for j = 1, batchSize do
        local w = word_o[j]
        if not ( (w == 0) ) then
          reward_mask[j][w] = 1
        end
      end
      if use_cuda then reward_mask = reward_mask:cuda() end
      gradOutputs[i] = gradOutputs[i]:cmul(reward_mask):clone()
      if use_cuda then word_i:cuda() gradOutputs[i]:cuda() nn.utils.recursiveType(qgen_input, 'torch.CudaTensor') end
      gradInputs[i] = qgen:backward(word_i, gradOutputs[i])
    end

    qgenW, qgendW = qgen:getParameters()
    qgendW:clamp(-5, 5)
    sgd(qgenW, qgendW, state.learningRate)
    qgen:zeroGradParameters()

    if retention then
      for p = 1, batchSize do
        if reward:squeeze()[p] > remeberThresh then
          pointer = pointer + 1
          trajectory_qas[pointer] = qas[p]:clone()
          trajectory_image_bgcolors[pointer] = train_image_bgcolors[t+p-1]:clone()
          trajectory_image_styles[pointer] = train_image_styles[t+p-1]:clone()
          trajectory_image_colors[pointer] = train_image_colors[t+p-1]:clone()
          trajectory_image_numbers[pointer] = train_image_numbers[t+p-1]:clone()
          trajectory_reward[pointer] = reward_target[p] + running_average
          trajectory_baseline[pointer] = running_average
          for s=1,seqLen do 
            local w = qas[p][s]
            if not (w == 0) then
              trajectory_prob[pointer][s] = outputs[s][p][w]
            end
          end
        end
      end
    end

  end
  print( 'epoch: '.. tostring(e) .. ' | Reward: ' .. tostring(running_average) )
  validating('valid')
  print( 'valid: '.. tostring(e) .. ' | Reward: ' .. tostring(running_average) )
  if best_valid <= running_average then
    torch.save(model_path, qgen:clearState()) 
    best_valid = running_average
    print('Saving the best model ...')
  end

  if retention then
    local earlyStop_retention = 0
    savepath_retention = 'retention_' .. model_path
    torch.save(savepath_retention, qgen:clearState())
    local prior_retention_average = running_average
    prior_retention_bestValid  = best_valid
    best_valid = running_average
    running_average_old = 0
    print("Saving current model before retention...")
    local draw = torch.floor(pointer/batchSize)
    local buffer_shuffle = torch.multinomial(torch.FloatTensor(draw*batchSize):fill(1), draw*batchSize, false)
    for rep = 1, repTresh do
      local sum_z = 0
      local count_z = 0
      running_average_z = 0
      qgen:clearState()
      qgen:training()
      for j=1, draw do
        xlua.progress(j, draw)
        local retention_image_bgcolors = torch.FloatTensor(batchSize, gridSize^2):fill(0)
        local retention_image_styles = torch.FloatTensor(batchSize, gridSize^2):fill(0)
        local retention_image_colors = torch.FloatTensor(batchSize, gridSize^2):fill(0)
        local retention_image_numbers = torch.FloatTensor(batchSize, gridSize^2):fill(0)
        local retention_qas = torch.FloatTensor(batchSize, (quesLen+1)*rounds):fill(0)
        local retention_reward = torch.FloatTensor(batchSize):fill(0)
        local retention_baseline = torch.FloatTensor(batchSize):fill(0)
        local retention_prob = torch.FloatTensor(batchSize, (quesLen+1)*rounds):fill(0)

        for p = (j-1)*batchSize+1, (j-1)*batchSize+batchSize do
          retention_image_bgcolors[(p-1)%batchSize+1] = trajectory_image_bgcolors[buffer_shuffle[p]]:clone()
          retention_image_styles[(p-1)%batchSize+1] = trajectory_image_styles[buffer_shuffle[p]]:clone()
          retention_image_colors[(p-1)%batchSize+1] = trajectory_image_colors[buffer_shuffle[p]]:clone()
          retention_image_numbers[(p-1)%batchSize+1] = trajectory_image_numbers[buffer_shuffle[p]]:clone()
          retention_qas[(p-1)%batchSize+1] = trajectory_qas[buffer_shuffle[p]]:clone()
          retention_reward[(p-1)%batchSize+1] = trajectory_reward[buffer_shuffle[p]]
          retention_baseline[(p-1)%batchSize+1] =  trajectory_baseline[buffer_shuffle[p]]
          retention_prob[(p-1)%batchSize+1] = trajectory_prob[buffer_shuffle[p]]:clone()
        end

        local p_log_prob = torch.FloatTensor(batchSize):fill(0)
        local q_log_prob = torch.FloatTensor(batchSize):fill(0)
        local z_weight = torch.FloatTensor(batchSize):fill(0)

        local qas_i = torch.FloatTensor(batchSize, rounds*(quesLen+1)):fill(0)
        local qas_o = torch.FloatTensor(batchSize, rounds*(quesLen+1)):fill(0)
        qas_i[{{},{1}}]:fill(wtoi['<START>'])
        qas_i[{{},{2,-1}}] = retention_qas[{{},{1,-2}}]:clone()
        qas_o[{{},{1,-1}}] = retention_qas:clone()

        qgen:training()
        qgen:zeroGradParameters()
        qgen:forget()

        local reward_target
        local outputs, reward_targets = {}, {}
        local seqLen = rounds*(quesLen+1)
        local qgen_input = { retention_image_bgcolors, retention_image_styles, retention_image_colors, retention_image_numbers }
        for i=1,seqLen do 
          local word_i = qas_i[{{},{i}}]:squeeze()
          local word_o = qas_o[{{},{i}}]:squeeze()
          if use_cuda then word_i:cuda() nn.utils.recursiveType(qgen_input, 'torch.CudaTensor') end
          outputs[i] = qgen:forward({word_i, unpack(qgen_input)})

          for k = 1, batchSize do
            p_log_prob[k] = p_log_prob[k] + torch.log(outputs[i][k][ word_o[k] ])
            q_log_prob[k] = q_log_prob[k] + torch.log(retention_prob[k][i])
            if updateBuffer then
              trajectory_prob[buffer_shuffle[(j-1)*batchSize+k]][i] = outputs[i][k][word_o[k]]
            end
          end
          if use_baseline2 then
            reward_target = (retention_reward - retention_baseline):clone()
          else
            reward_target = retention_reward:clone()
          end
          reward_targets[i] = reward_target:view(reward_target:size(1), 1):clone()
          if use_cuda then outputs[i]:cuda() reward_targets[i] = reward_targets[i]:cuda() end
          criterion_qgen:forward(outputs[i], reward_targets[i])
        end

        z_weight = torch.exp(p_log_prob - q_log_prob)   

        if not (trustFactor == 0) then
          for i=1, z_weight:size(1) do
            if (z_weight[i] > trustFactor) or (z_weight[i] < trustFactor_inverse) then
              z_weight[i] = 0
            end
          end
        end

        sum_z = sum_z + z_weight:ne(0):sum()
        count_z = count_z + 1
        running_average_z = torch.floor((sum_z/count_z/batchSize)*10000)/100

        local gradOutputs, gradInputs = {}, {}
        for i=seqLen,1,-1 do 
          local word_i = qas_i[{{},{i}}]:squeeze()
          local word_o = qas_o[{{},{i}}]:squeeze()

          if use_cuda then outputs[i]:cuda() reward_targets[i] = reward_targets[i]:cuda() end
          gradOutputs[i] = criterion_qgen:backward(outputs[i], reward_targets[i])
          local reward_mask = torch.FloatTensor(gradOutputs[i]:size()):fill(0)
          for j = 1, batchSize do
            local w = word_o[j]
            if not ( (w == 0) ) then
              reward_mask[j][w] = 1
            end
          end
          local z_weight_repeat = torch.repeatTensor(z_weight:float():view(gradOutputs[i]:size(1), 1), 1, gradOutputs[i]:size(2))
          if use_cuda then reward_mask = reward_mask:cuda() z_weight_repeat = z_weight_repeat:cuda() end
          gradOutputs[i] = gradOutputs[i]:cmul(reward_mask):clone()
          gradOutputs[i] = torch.cmul(gradOutputs[i], z_weight_repeat):clone() --debug
          for k, v in ipairs(qgen:get(1).forwardnodes) do
            if v.data.annotations.name == 'output' then
              v.data.module._input = outputs[i]:clone()
            end
          end

          if use_cuda then word_i:cuda() gradOutputs[i]:cuda() nn.utils.recursiveType(qgen_input, 'torch.CudaTensor') end
          gradInputs[i] = qgen:backward(word_i, gradOutputs[i])
        end

        qgenW, qgendW = qgen:getParameters()
        qgendW:clamp(-5, 5)
        sgd(qgenW, qgendW, state.learningRate)
        qgen:zeroGradParameters()
      end

      validating( 'valid' )
      print( 'retention: '.. tostring(rep) .. ' | Reward: ' .. tostring(running_average) )
      if best_valid <= running_average then
        best_valid = running_average
        earlyStop_retention = 0
        torch.save(savepath_retention, qgen:clearState())
        print("Saving replyed model...")
        if prior_retention_bestValid <= running_average then
          torch.save(model_path, qgen:clearState()) 
          print('Saving the best model ...')
        end
      else
        earlyStop_retention = (earlyStop_retention + 1)
      end

      if running_average >= running_average_old then
        running_average_old = running_average
        earlyStop_retention = 0
      end

      if earlyStop_retention >= earlyStopThresh then
        print("Retention early stopped at epoch: " .. tostring(rep))
        break
      end
    end
    trajectory_qas = torch.FloatTensor(trainSize, (quesLen+1)*rounds):fill(0)
    trajectory_image_bgcolors = torch.FloatTensor(trainSize, gridSize^2):fill(0)
    trajectory_image_styles = torch.FloatTensor(trainSize, gridSize^2):fill(0)
    trajectory_image_colors = torch.FloatTensor(trainSize, gridSize^2):fill(0)
    trajectory_image_numbers = torch.FloatTensor(trainSize, gridSize^2):fill(0)
    trajectory_reward = torch.FloatTensor(trainSize):fill(0)
    trajectory_prob = torch.FloatTensor(trainSize, (quesLen+1)*rounds):fill(0)
    pointer = 0
    best_valid = math.max(prior_retention_bestValid, best_valid)
  end
end

qgenW, qgendW = qgen:getParameters()
if not (epochs == 0) then qgenW:copy( torch.load(model_path):getParameters() ) end
validating('test')
print( 'test: '.. tostring(epochs) .. ' | Reward: ' .. tostring(running_average) )

