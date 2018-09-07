require 'optim'
require 'rnn'
require 'hdf5'
require 'xlua'
require 'nngraph'
cjson = require 'cjson'
utils = require 'misc.utils'
dofile("misc/optim_updates.lua")
dofile("misc/vocab.lua")

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(123)

fd = io.open('sample.txt', 'a')
fd:write('\n'..'******************* '..os.date("%c")..' *******************'..'\n')
fd:close()

quesLen = 7
batchSize = 2
epochs = 30
use_cuda = false
deviceId = 1
rounds = 8

state = {}
state.learningRate = 0.001

load_model = true
update_weight = false

if use_cuda then 
  require 'cunn' 
  require 'cutorch'
  print("Using GPU:"..deviceId)
  cutorch.setDevice(deviceId)
  cutorch.setHeapTracking(true)
  cutorch.manualSeed(123)
end

running_average = 0
running_average_z = 0
long_qa = 0

trainSize = 0 
for line in io.lines('data/train.json') do 
  local example = cjson.decode(line)
  trainSize = trainSize + 1 
  gridSize = #example['gridImg'] 
end

bgcolors = torch.FloatTensor(trainSize, gridSize^2):fill(0)
styles = torch.FloatTensor(trainSize, gridSize^2):fill(0)
colors = torch.FloatTensor(trainSize, gridSize^2):fill(0)
numbers = torch.FloatTensor(trainSize, gridSize^2):fill(0)
qas_i = torch.FloatTensor(trainSize, (quesLen+1)*rounds+1):fill(0)
qas_o = torch.FloatTensor(trainSize, (quesLen+1)*rounds+1):fill(0)

local i = 1
for line in io.lines('data/train.json') do
  local example = cjson.decode(line)
  for x = 1, gridSize do 
    for y = 1, gridSize do
      cell = example['gridImg'][x][y]
      bgcolors[i][(x-1)*gridSize+y] = bgcolorstoi[cell['bgcolor']]
      styles[i][(x-1)*gridSize+y] = stylestoi[cell['style']]
      colors[i][(x-1)*gridSize+y] = colorstoi[cell['color']]
      numbers[i][(x-1)*gridSize+y] = cell['number']+1
    end
  end
  qa_s = {}
  table.insert(qa_s, '<START>')
  for q = 1, #example['qa'] do
    if q < (rounds+1) then
      local qa = string.split(example['qa'][q]['question']," +")
      table.insert(qa, example['qa'][q]['answer'])
      for w = 1, #qa do 
        qa_s[#qa_s+1] = qa[w]
      end
    else
      long_qa = long_qa + 1
      break
    end
  end

  local len = #qa_s

  for w = 1, len do
    qas_i[i][(quesLen+1)*rounds+1-len+w] = wtoi[qa_s[w]]
    if w < len then
    qas_o[i][(quesLen+1)*rounds+1-len+w] = wtoi[qa_s[w+1]]
    else
      qas_o[i][(quesLen+1)*rounds+1-len+w] = wtoi['<END>']
    end
  end
  i = i + 1
end

print('trainSize', trainSize, 'gridSize', gridSize)

shuffle = torch.randperm(trainSize)

dofile("misc/qgen.lua")

output =
  rnn_cell
- nn.Linear(hiddenSize, #itow)
- nn.LogSoftMax()

output:annotate{name = 'output'}

table.insert(outputs, output)

model = nn.gModule(inputs, outputs)
model = nn.Recursor(model, histLength)
model = require('misc.weight-init')(model, 'xavier')

criterion = nn.ClassNLLCriterion()
criterion.ignoreIndex = 0
criterion.sizeAverage = false

if load_model then
  modelW, modeldW = model:getParameters()
  modelW:copy( torch.load('qgen_float.t7'):getParameters() )
end

if use_cuda then model:cuda() criterion:cuda() end

function sampling(input, seqLen)
  model:evaluate()
  model:forget()
  local word_i = input[1]:clone():fill(wtoi['<START>'])[{{}, {1}}]:squeeze()
  local outputs = {}

  for i=1,seqLen do
    outputs[i] = model:forward({word_i, input[2], input[3], input[4], input[5]})
    local nextToken = torch.multinomial(torch.exp(outputs[i] / 1), 1)
    local word = nextToken:float()[{{1},{1}}] 

    fd = io.open('sample.txt', 'a')
    fd:write(utils.vec2str(word))
    fd:close()
    word_i = nextToken:squeeze()
  end
  fd = io.open('sample.txt', 'a')
  fd:write('\n')
  fd:close()
end

model_path = 'qgen '..os.date("%c") .. '.t7'

for e = 1, epochs do
  running_average = 0
  curLoss = 0
  numTokens = 0
  for t = 1, torch.floor(trainSize/batchSize)*batchSize, batchSize do
    xlua.progress(torch.floor(t/batchSize)+1, torch.floor(trainSize/batchSize))
    model:training()
    model:forget()

    local input = { qas_i[{{t, t+batchSize-1}}], bgcolors[{{t, t+batchSize-1}}], styles[{{t, t+batchSize-1}}], 
    colors[{{t, t+batchSize-1}}], numbers[{{t, t+batchSize-1}}] }
    local label = qas_o[{{t, t+batchSize-1}}]
    if use_cuda then nn.utils.recursiveType(input, 'torch.CudaTensor') label:cuda() end

    local seqLen, Len= 0, 0
    for j = t, (t+batchSize-1) do
      Len = qas_i[j]:ne(0):sum()
      if seqLen < Len then
        seqLen = Len
      end
    end

    local input_seq  = input[1][{{},{(quesLen+1)*rounds+1-seqLen+1, (quesLen+1)*rounds+1}}]:clone()
    local output_seq = label[{{},{(quesLen+1)*rounds+1-seqLen+1, (quesLen+1)*rounds+1}}]:clone()

    local outputs = {}

    for i=1,seqLen do 
      local word_i = input_seq[{{},{i}}]:squeeze()
      local word_o = output_seq[{{},{i}}]:squeeze()
      outputs[i] = model:forward({word_i, input[2], input[3], input[4], input[5]})
      local _, nextToken = torch.max(torch.exp(outputs[i] / 1), 2)
      local word = nextToken:float()[{{1},{1}}] 

      curLoss = curLoss + criterion:forward(outputs[i], word_o)
    end

    local gradOutputs, gradInputs = {}, {}
    for i=seqLen,1,-1 do 
      local word_i = input_seq[{{},{i}}]:squeeze()
      local word_o = output_seq[{{},{i}}]:squeeze()
      gradOutputs[i] = criterion:backward(outputs[i], word_o)
      gradInputs[i] = model:backward({word_i, input[2], input[3], input[4], input[5]}, gradOutputs[i])
    end

    numTokens = numTokens + torch.sum(label:gt(0))
    running_average = curLoss/(numTokens+0.00000000001)

    modelW, modeldW = model:getParameters()
    if update_weight then
      adam(modelW, modeldW, state)
      model:zeroGradParameters()
    end

    input_seq = nil
    output_seq = nil

    sampling(input, seqLen)
  end
  torch.save(model_path, model:clearState()) 
  print( 'epoch: '.. tostring(e) .. ' | loss: ' .. tostring(running_average) )
end

