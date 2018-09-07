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

if use_cuda then 
  require 'cunn' 
  require 'cutorch'
  print("Using GPU:"..deviceId)
  cutorch.setDevice(deviceId)
  cutorch.setHeapTracking(true)
  cutorch.manualSeed(123)
end

trainSize = 0 
for line in io.lines('data/train.json') do 
  local example = cjson.decode(line)
  for i = 1, #example['qa'] do
    trainSize = trainSize + 1 
  end
end

bgcolors = torch.FloatTensor(trainSize):fill(0)
styles = torch.FloatTensor(trainSize):fill(0)
colors = torch.FloatTensor(trainSize):fill(0)
numbers = torch.FloatTensor(trainSize):fill(0)
answers = torch.FloatTensor(trainSize):fill(0)
questions = torch.FloatTensor(trainSize, quesLen):fill(0)

local i = 1
for line in io.lines('data/train.json') do
  local example = cjson.decode(line)
  cell = example['gridImg'][example['target'][1]+1][example['target'][2]+1]
  bgcolors[i] = bgcolorstoi[cell['bgcolor']]
  styles[i] = stylestoi[cell['style']]
  colors[i] = colorstoi[cell['color']]
  numbers[i] = cell['number']+1

  for q = 1, #example['qa'] do
    answers[i] = answertoi[example['qa'][q]['answer']]
    local question = string.split(example['qa'][q]['question']," +")
    local len = #question
    for w = 1, len do
      questions[i][quesLen-len+w] = wtoi[question[w]]
    end
    i = i + 1
  end
end

inputs = {}
outputs = {}

table.insert(inputs, nn.Identity()()) -- ques
table.insert(inputs, nn.Identity()()) -- bgcolor
table.insert(inputs, nn.Identity()()) -- style
table.insert(inputs, nn.Identity()()) -- color
table.insert(inputs, nn.Identity()()) -- number

ques = inputs[1]
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

model_ques =  
  ques
- nn.Transpose({1,2})
- nn.LookupTableMaskZero(#itow, hiddenSize)
- rnn
- nn.Select(1, -1)

model_bgcolor =
  bgcolor
- nn.LookupTableMaskZero(lookup_bgcolor, hiddenSize/4)
- nn.ReLU()
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()

model_style =
  style
- nn.LookupTableMaskZero(lookup_style, hiddenSize/4)
- nn.ReLU()
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()

model_color =
  color
- nn.LookupTableMaskZero(lookup_color, hiddenSize/4)
- nn.ReLU()
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()

model_number =
  number
- nn.LookupTableMaskZero(lookup_number, hiddenSize/4)
- nn.ReLU()
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()

model_main = 
  { model_ques, model_bgcolor, model_style, model_color, model_number } 
- nn.JoinTable(2)
- nn.Linear(2*hiddenSize, 2*hiddenSize)
- nn.ReLU()
- nn.Linear(2*hiddenSize, 2)
- nn.LogSoftMax()
   
table.insert(outputs, model_main)
model = nn.gModule(inputs, outputs)
criterion = nn.ClassNLLCriterion()
if use_cuda then model:cuda() criterion:cuda() end
confusion = optim.ConfusionMatrix(2)
state = {}
state.learningRate = 0.001

running_average = 0
running_average_z = 0

model_path = 'oracle '..os.date("%c") .. '.t7'

for i = 1, epochs do
  confusion:zero()
  for t = 1, torch.floor(trainSize/batchSize)*batchSize, batchSize do
    xlua.progress(torch.floor(t/batchSize)+1, torch.floor(trainSize/batchSize))
    model:training()
    local input = { questions[{{t, t+batchSize-1}}], bgcolors[{{t, t+batchSize-1}}], styles[{{t, t+batchSize-1}}], 
    colors[{{t, t+batchSize-1}}], numbers[{{t, t+batchSize-1}}] }
    local label = answers[{{t, t+batchSize-1}}]
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
