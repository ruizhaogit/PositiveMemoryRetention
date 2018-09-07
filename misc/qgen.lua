inputs = {}
outputs = {}

table.insert(inputs, nn.Identity()()) -- qa_i
table.insert(inputs, nn.Identity()()) -- bgcolor
table.insert(inputs, nn.Identity()()) -- style
table.insert(inputs, nn.Identity()()) -- color
table.insert(inputs, nn.Identity()()) -- number

qa_i = inputs[1]
bgcolor = inputs[2]
style = inputs[3]
color = inputs[4]
number = inputs[5]

lookup_bgcolor = 5
lookup_style = 2
lookup_color = 5
lookup_number = 10

hiddenSize = 256

nn.FastLSTM.usenngraph = true
rnn = nn.FastLSTM(hiddenSize*2, hiddenSize)
rnn:maskZero(1)

bgcolor_embed =
  bgcolor
- nn.LookupTableMaskZero(lookup_bgcolor, hiddenSize/4)
- nn.ReLU()
- nn.View(-1,hiddenSize/4)
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()
- nn.View(-1, gridSize^2, hiddenSize/4)

style_embed =
  style
- nn.LookupTableMaskZero(lookup_style, hiddenSize/4)
- nn.ReLU()
- nn.View(-1,hiddenSize/4)
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()
- nn.View(-1, gridSize^2, hiddenSize/4)

color_embed =
  color
- nn.LookupTableMaskZero(lookup_color, hiddenSize/4)
- nn.ReLU()
- nn.View(-1,hiddenSize/4)
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()
- nn.View(-1, gridSize^2, hiddenSize/4)

number_embed =
  number
- nn.LookupTableMaskZero(lookup_number, hiddenSize/4)
- nn.ReLU()
- nn.View(-1,hiddenSize/4)
- nn.Linear(hiddenSize/4, hiddenSize/4)
- nn.ReLU()
- nn.View(-1, gridSize^2, hiddenSize/4)

multi_embed = 
  { bgcolor_embed, style_embed, color_embed, number_embed }
- nn.CAddTable()
- nn.View(-1,(hiddenSize/4)*gridSize^2)
- nn.MaskZero(nn.Linear((hiddenSize/4)*gridSize^2, hiddenSize*2), 1)
- nn.ReLU()
- nn.MaskZero(nn.Linear(hiddenSize*2, hiddenSize), 1)
- nn.ReLU()

word_embed = 
  qa_i
- nn.LookupTableMaskZero(#itow, hiddenSize)
- nn.Squeeze()

both_embed = 
  {word_embed, multi_embed}
- nn.JoinTable(2)

rnn_cell =
  both_embed
- rnn

rnn_cell:annotate{name = 'rnn_cell'}