require 'nn'
require 'optim'
require 'itemItemDataLeo'
require 'nngraph'
require 'rnn'
require 'util'
require 'utilExp'
require 'lfs'
local inspect = require 'inspect'

START_EPOCH = 1
OUTPUT_DIR = '../output/leo/'
LEARNING_RATES = {30, 30, 30, 10, 10, 10, 5, 5, 5}
LEARNING_RATE_REPEATS = 4
MIN_LEARNING_RATE = 1

local fileId = arg[1]
assert(fileId ~= nil)

local modelDir = nil
if(arg[2] ~= nil) then
  START_EPOCH = arg[2]
  modelDir = '../output/' .. fileId .. '/models/' .. fileId .. '_' .. START_EPOCH
  --print(modelDir)
end

math.randomseed(os.time())

data = DataAssistMatrix()
collectgarbage()

local n_hidden = 200
local decay_rate = 1 
local init_rate = 30
local mini_batch_size = 60
local dropoutPred = true
local max_grad = 5e-5 

local rnn = RNN{
	dropoutPred = dropoutPred,
	n_hidden = n_hidden,
	n_questions = data.n_questions,
	maxGrad = max_grad,
	maxSteps = 2,
	compressedSensing = true,
	compressedDim = 100,
  modelDir = modelDir
}

local batch = data:getTestBatch()
--rnn:fprop2(batch)

--os.exit()
local allPredictions = {}
local totalPositives = 0
local totalNegatives = 0

local fi = 1; -- from item
local fo = 0; -- from outcome
local ti = 1; -- to item
local pps = rnn:getPredictionTruth(batch)
for i,prediction in ipairs(pps) do
  print(i .. '\t' .. fi .. '\t' .. fo .. '\t' .. ti .. '\t' .. prediction['truth'] .. '\t' .. prediction['pred'])

  if(fo == 0) then
    fo=1
  elseif(ti < 30) then
    ti = ti + 1
    fo = 0
  else
    fi = fi + 1
    ti = 1
    fo = 0
  end

  if(prediction['truth'] == 1) then
		totalPositives = totalPositives + 1
	else
		totalNegatives = totalNegatives + 1
	end

	table.insert(allPredictions, prediction)
end
os.exit()
collectgarbage()

function compare(a,b)
	return a['pred'] > b['pred']
end
table.sort(allPredictions, compare)

local truePositives = 0
local falsePositives = 0
local correct = 0
local auc = 0
local lastFpr = nil
local lastTpr = nil
for i,p in ipairs(allPredictions) do
	if(p['truth'] == 1) then
		truePositives = truePositives + 1
	else
		falsePositives = falsePositives + 1
	end

	local guess = 0
	if(p['pred'] > 0.5) then guess = 1 end
	if(guess == p['truth']) then correct = correct + 1 end
	local fpr = falsePositives / totalNegatives
	local tpr = truePositives / totalPositives
	if(i % 500 == 0) then
		if lastFpr ~= nil then
			local trapezoid = (tpr + lastTpr) * (fpr - lastFpr) *.5
			auc = auc + trapezoid
		end	
		lastFpr = fpr
		lastTpr = tpr
	end
	if(recall == 1) then break end
end

local accuracy = correct / #allPredictions
print(auc)
print(accuracy)

