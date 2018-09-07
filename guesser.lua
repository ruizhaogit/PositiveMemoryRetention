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

quesLen = 7
batchSize = 256
epochs = 30
use_cuda = true
deviceId = 1
rounds = 8

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
labels = torch.FloatTensor(trainSize):fill(0)
qas = torch.FloatTensor(trainSize, (quesLen+1)*rounds):fill(0)

local i = 1
for line in io.lines('data/train.json') do
  local example = cjson.decode(line)
  labels[i] = example['target'][1]*gridSize+example['target'][2]+1
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
  for q = 1, #example['qa'] do
    if q < (rounds+1) then
      local qa = string.split(example['qa'][q]['question']," +")
      table.insert(qa, example['qa'][q]['answer'])
      for w = 1, #qa do 
        qa_s[#qa_s+1] = qa[w]
      end

      local len = #qa_s
      for w = 1, len do
        qas[i][(quesLen+1)*rounds-len+w] = wtoi[qa_s[w]]
      end
    else
      long_qa = long_qa + 1
      break
    end
  end
  i = i + 1
end

print('trainSize', trainSize, 'gridSize', gridSize)

inputs = {}
outputs = {}

table.insert(inputs, nn.Identity()()) -- dial
table.insert(inputs, nn.Identity()()) -- bgcolor
table.insert(inputs, nn.Identity()()) -- style
table.insert(inputs, nn.Identity()()) -- color
table.insert(inputs, nn.Identity()()) -- number

dial = inputs[1]
bgcolor = inputs[2]
style = inputs[3]
color = inputs[4]
number = inputs[5]

lookup_bgcolor = 5
lookup_style = 2
lookup_color = 5
lookup_number = 10

hiddenSize = 64

rnn = nn.SeqLSTM(hiddenSize, hiddenSize)
rnn.maskzero = true 

model_dial =  
  dial
- nn.Transpose({1,2})
- nn.LookupTableMaskZero(#itow, hiddenSize)
- rnn
- nn.Select(1, -1)
- nn.Replicate(gridSize^2, 2)

mlp_bgcolor =  nn.Sequential()
mlp_bgcolor:add(nn.LookupTableMaskZero(lookup_bgcolor, hiddenSize/4))
mlp_bgcolor:add(nn.ReLU())
mlp_bgcolor:add(nn.View(-1,hiddenSize/4))
mlp_bgcolor:add(nn.Linear(hiddenSize/4, hiddenSize/4))
mlp_bgcolor:add(nn.ReLU())
mlp_bgcolor:add(nn.Reshape(1, hiddenSize/4), true)

model_mlp_bgcolor =
  bgcolor
- nn.Contiguous()
- nn.View(-1, gridSize^2, 1)
- nn.SplitTable(1, 2)
- nn.MapTable(mlp_bgcolor)
- nn.JoinTable(1, 2)

mlp_style =  nn.Sequential()
mlp_style:add(nn.LookupTableMaskZero(lookup_style, hiddenSize/4))
mlp_style:add(nn.ReLU())
mlp_style:add(nn.View(-1,hiddenSize/4))
mlp_style:add(nn.Linear(hiddenSize/4, hiddenSize/4))
mlp_style:add(nn.ReLU())
mlp_style:add(nn.Reshape(1, hiddenSize/4), true)

model_mlp_style =
  style
- nn.Contiguous()
- nn.View(-1, gridSize^2, 1)
- nn.SplitTable(1, 2)
- nn.MapTable(mlp_style)
- nn.JoinTable(1, 2)

mlp_color =  nn.Sequential()
mlp_color:add(nn.LookupTableMaskZero(lookup_color, hiddenSize/4))
mlp_color:add(nn.ReLU())
mlp_color:add(nn.View(-1,hiddenSize/4))
mlp_color:add(nn.Linear(hiddenSize/4, hiddenSize/4))
mlp_color:add(nn.ReLU())
mlp_color:add(nn.Reshape(1, hiddenSize/4), true)

model_mlp_color =
  color
- nn.Contiguous()
- nn.View(-1, gridSize^2, 1)
- nn.SplitTable(1, 2)
- nn.MapTable(mlp_color)
- nn.JoinTable(1, 2)

mlp_number =  nn.Sequential()
mlp_number:add(nn.LookupTableMaskZero(lookup_number, hiddenSize/4))
mlp_number:add(nn.ReLU())
mlp_number:add(nn.View(-1,hiddenSize/4))
mlp_number:add(nn.Linear(hiddenSize/4, hiddenSize/4))
mlp_number:add(nn.ReLU())
mlp_number:add(nn.Reshape(1, hiddenSize/4), true)

model_mlp_number =
  number
- nn.Contiguous()
- nn.View(-1, gridSize^2, 1)
- nn.SplitTable(1, 2)
- nn.MapTable(mlp_number)
- nn.JoinTable(1, 2)

mlp = nn.Sequential()
mlp:add(nn.Linear(hiddenSize, hiddenSize))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(hiddenSize, hiddenSize))
mlp:add(nn.ReLU())

split_mlp = nn.Sequential()
split_mlp:add(nn.Reshape(2, hiddenSize), true)
split_mlp:add(nn.SplitTable(1,2))
split_mlp:add(nn.ParallelTable():add(nn.Identity()):add(mlp))
split_mlp:add(nn.JoinTable(2))
split_mlp:add(nn.Reshape(1, hiddenSize*2), true)

dot = nn.Sequential()
dot:add(nn.Reshape(2, hiddenSize))
dot:add(nn.SplitTable(1,2))
dot:add(nn.DotProduct())
dot:add(nn.View(-1, 1))

model_main = 
  { model_dial, model_mlp_bgcolor, model_mlp_style, model_mlp_color, model_mlp_number }
- nn.JoinTable(3)
- nn.SplitTable(1,2)
- nn.MapTable(split_mlp)
- nn.JoinTable(2)
- nn.SplitTable(1,2)
- nn.MapTable(dot) 
- nn.JoinTable(2)
- nn.LogSoftMax()
   
table.insert(outputs, model_main)
model = nn.gModule(inputs, outputs)
criterion = nn.ClassNLLCriterion()
if use_cuda then model:cuda() criterion:cuda() end
confusion = optim.ConfusionMatrix(gridSize^2)
state = {}
state.learningRate = 0.01

running_average = 0

model_path = 'guesser '..os.date("%c") .. '.t7'

for i = 1, epochs do
  confusion:zero()
  for t = 1, torch.floor(trainSize/batchSize)*batchSize, batchSize do
    xlua.progress(torch.floor(t/batchSize)+1, torch.floor(trainSize/batchSize))
    model:training()
    local input = { qas[{{t, t+batchSize-1}}], bgcolors[{{t, t+batchSize-1}}], styles[{{t, t+batchSize-1}}], 
    colors[{{t, t+batchSize-1}}], numbers[{{t, t+batchSize-1}}] }
    local label = labels[{{t, t+batchSize-1}}]
    if use_cuda then nn.utils.recursiveType(input, 'torch.CudaTensor') label:cuda() end
    local output = model:forward(input)
    local loss = criterion:forward(output, label)
    local gradOut = criterion:backward(output, label)
    model:backward(input, gradOut)
    modelW, modeldW = model:getParameters()
    modeldW:clamp(-5, 5)
    adam(modelW, modeldW, state)
    model:zeroGradParameters()
    confusion:batchAdd(output, label)
    confusion:updateValids()
    running_average = confusion.totalValid
  end
  torch.save(model_path, model:clearState())
  print(confusion.totalValid)
end
