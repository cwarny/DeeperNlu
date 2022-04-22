# DeeperNlu

## Installation

```
git clone ssh://git.amazon.com/pkg/DeeperNlu
cd DeeperNlu
python setup.py install
```

DeeperNlu has some dependencies that are not available in the standard PyTorch DL AMI. In particular, the package `transformers`, very useful for all things BERT and other things, needs to be installed manually. Here are the steps:

* `source activate pytorch_p36`
* `pip install -r requirements.txt`

## Straightforward data ingestion pipeline

### An example

```
from deeper_nlu.data import TabularDataset
from deeper_nlu.data.processor import TokenizeProcessor, NumericalizeProcessor, CategoryProcessor

tfms = {
    'intent': lambda d: '_'.join([d['domain'], d['intent']]),
    'text': lambda d: ' '.join([item.split('|')[0] for item in d['annotation'].split(' ')]),
    'labels': lambda d: ' '.join([item.split('|')[1] for item in d['annotation'].split(' ')])
}

proc_x = {
    'text': [TokenizeProcessor(), NumericalizeProcessor()],
    'profile': CategoryProcessor(min_freq=0)
}

proc_y = {
    'intent': CategoryProcessor(min_freq=0),
    'labels': [TokenizeProcessor(), NumericalizeProcessor(min_freq=0)]
}

data = TabularDataset \
    .from_csv('data.tsv', names=['domain', 'intent', 'annotation', 'profile'], tfms=tfms) \
    .split(stratify_by='domain') \
    .label(proc_x=proc_x, proc_y=proc_y) \
    .to_databunch()
```

The above creates a `DataBunch` object that holds two data loaders: one for the training data and one for the validation data. The pipeline took a CSV file, split it into training and validation in a stratified manner, processed both input and target variables appropriately (tokenizing and numericalizing the sequence input and sequence target, categorizing the profile input and the intent target), and then stored the result in a `DataBunch` object that takes care of batching the data and collating each batch.

### Transformers and processors

Two important concepts here: transformers and processors. Transformers are applied on the fields of the input data to do any kind of preprocessing on those fields or to create new fields. In the above, we modified the `intent` field and created two new fields: `text` and `labels`, based on the other fields. A transformer is just a dictionary mapping the name of the field created or modified to a function that takes a dict-formatted row of the dataset as an argument.

A processor processes the input data to put it in the right format for doing machine learning. That usually involves at some point turning strings into numbers. The main difference with transformers is that processors typically maintain some state. Common processors in NLP include tokenizers and numericalizers, but it's very easy to create your own processor by subclassing the base `Processor` class. You will need to define at least two methods: `proc1`, and `deproc1`, the former implementing how a single sample should be processed, the latter how it should be "deprocessed". You can use the class constructor `__init__` to initialize whatever state your processor needs. Deprocessing is typically useful during inference as we will see. A processor, while an object, is applied to an input field like a function, meaning the special method `__call__` will be called with the input as argument. By default, the `__call__` method will call the `process` method, which in turn simply calls `proc1` over each item in the input. Let's look at an example:

```
class CategoryProcessor(Processor):
    def __init__(self, vocab=None, min_freq=2, default_category='Other'):
        self.vocab, self.min_freq, self.default_category = vocab, min_freq, default_category
    
    def __call__(self, items):
        if self.vocab is None:
            freq = Counter(o for o in items)
            self.vocab = [o for o,c in freq.most_common() if c >= self.min_freq]
            self.vocab.insert(0, self.default_category)
        if getattr(self, 'otoi', None) is None:
            self.otoi  = defaultdict(int, {v:k for k,v in enumerate(self.vocab)}) # Returns 0 if can't find
        return self.process(items)
    
    def proc1(self, item): 
        return self.otoi[item]
    
    def deproc1(self, idx):
        return self.vocab[idx]
```

The above processor receives a categorical field as input. The state here simply consists in keep two mappings: one mapping a string to an index, and vice-versa. You can either pass a pre-built vocab in the constructor, or it will be created and added to the processor's state the first time it is called. We do that by overwriting the `__call__` method. We define `proc1` as simply looking up the index of a category string, and `deproc1` as looking up the string corresponding to some index. It's as simple as that!

Processors are passed to the `label` method of a dataset object. The `label` method distinguishes between processors for the inputs to our model and processors for the targets of our model:

```
data = TabularDataset.from_csv(...) \
    .label(proc_x={'text': [TokenizeProcessor(), NumericalizeProcessor()]}, proc_y={'intent':CategoryProcessor()})
```

Like for transformers, we need to specify which field a given processor or series of processors need to be applied to, by using a dictionary mapping a field name to its processor(s). Some fields might need multiple processors applied one after the other (e.g., text fields), just map that field name to a list of processors, and those will be composed.

### High flexibility

To further show the flexibility of this data pipeline, consider how it could be used for a different use case. Let's say you want to create a model that mimics DIP. Your input data is AMDS-annotated utterances and your target is those same utterances but annotated in NLU space. Your input file has six columns: AMDS domain, AMDS intent, AMDS annotation, NLU domain, NLU intent, and finally NLU annotation. Here is how we would write the data pipeline for this:

```
tfms = {
    'text': lambda d: ' '.join([item.split('|')[0] for item in d['amds_annotation'].split()]),
    'amds_intent': lambda d: '_'.join([d['amds_domain'], d['amds_intent']]) if d['amds_domain'] else '',
    'amds_labels': lambda d: ' '.join([item.split('|')[1] for item in d['amds_annotation'].split()]),
    'nlu_intent': lambda d: '_'.join([d['nlu_domain'], d['nlu_intent']]) if d['nlu_domain'] else '',
    'nlu_labels': lambda d: ' '.join([item.split('|')[1] for item in d['nlu_annotation'].split()])
}

proc_x = {
    'text': [TokenizeProcessor(), NumericalizeProcessor()],
    'amds_intent': CategoryProcessor(min_freq=0),
    'amds_labels': [TokenizeProcessor(), NumericalizeProcessor(min_freq=0)]
}

proc_y = {
    'nlu_intent': CategoryProcessor(min_freq=0),
    'nlu_labels': [TokenizeProcessor(), NumericalizeProcessor(min_freq=0)]
}

data = TabularDataset \
    .from_csv('data/sample.tsv.parallel', names=['amds_domain', 'amds_intent', 'amds_annotation', 'nlu_domain', 'nlu_intent', 'nlu_annotation', 'cust_id', 'utt_id'], tfms=tfms) \
    .split(stratify_by='amds_domain') \
    .label(proc_x=proc_x, proc_y=proc_y) \
    .to_databunch()
```

Super easy! Very few changes are needed even though the data looks quite different.

## Infinitely customizable training loop

Borrowing directly from [fast.ai](fast.ai)'s API, training in DeeperNlu is managed by the `Learner` class, which exposes callbacks at different stages of the training loop. You can easily inject callbacks to be run at any of the following events, having full access to all the main objects exposed by the learner:

* `begin_batch`
* `after_pred`
* `after_loss`
* `after_backward`
* `after_step`
* `after_cancel_batch`
* `after_batch`
* `after_cancel_epoch`
* `begin_fit`
* `begin_epoch`
* `begin_validate`
* `after_epoch`
* `after_cancel_train`
* `after_fit`

Define a custom callback by extending the base `Callback` class. For example, a callback to switch between training and validating:

```
from DeeperNlu.train.callback import Callback

class TrainEvalCallback(Callback):
    _order = 0
    def begin_fit(self):
        self.run.n_epochs = 0.
        self.run.n_iter = 0
    
    def after_batch(self):
        if not self.in_train: return # don't do anything
        self.run.n_epochs += 1./self.iters
        self.run.n_iter += 1
    
    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train = True
    
    def begin_validate(self):
        self.model.eval()
        self.run.dl = self.data.valid_dl
        self.run.in_train = False
```

Instances of the `Callback` class forward access to an attribute ("`getattr` events") to the learner if it can't find it at the level of its own instance. So `self.in_train` or `self.model` are forwarded to the learner. To *set* an attribute on the learner, use the special property `self.run` which refers to the learner.

Another example of a callback for early stopping:

```
from DeeperNlu.train.callback import Callback

class EarlyStopCallback(Callback):
    def after_step(self):
        if self.n_iter >= 10:
            raise CancelTrainException()
```

How to inject those callbacks into the training loop? Inject them when instantiating the learner. Here is a typical usage, with additional callbacks:

```
from DeeperNlu.loss import CombinedLoss
from DeeperNlu.nn import BertForIcAndNer
from DeeperNlu.metric import combined_accuracy, class_accuracy, seq_accuracy
from DeeperNlu.train import Learner
from DeeperNlu.train.callback import AvgStatsCallback, CudaCallback, SaveModel
from torch.optim import Adam

model = BertForIcAndNer(encoding_size, label_vocab_size, intent_vocab_size)

loss_func = CombinedLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cbfs = [
    partial(AvgStatsCallback, [combined_accuracy, class_accuracy, seq_accuracy]), 
    partial(CudaCallback, device),
    partial(SaveModel, path/'model.pth')
]
lr = 1e-3
opt = partial(Adam, lr=lr, weight_decay=1e-3)
learn = Learner(model, data, loss_func, lr=lr, cb_funcs=cbfs, opt_func=opt)
```

After instantiating the learner, fitting the model is as simple as the following:

```
n_epochs = 50
learn.fit(n_epochs)
```

The `AvgStatsCallback` callback will print out loss and all accuracy metrics to the terminal as training happens, the `CudaCallback` callback will ensure that computation happens on the desired device, and the `SaveModel` callback ensure the model is saved when improvements on validation loss happen.

## Inference

Once you've trained your model, it's super easy to test it on some held-out labelled test set, or to predict on new unlabelled data.

This is how you would load a test set:

```
test_data = TabularDataset \
    .from_csv('data/test.tsv', names=['domain', 'intent', 'annotation', 'profile'], tfms=tfms) \
    .split(train_proportion=1.) \
    .label(proc_x=proc_x, proc_y=proc_y) \
    .to_databunch()
```

Load your model from saved weights, create a learner, and run `fit` for one epoch:

```
model = Model(*args, **kwargs)
model.load_state_dict(torch.load('model.pth'))
loss_func = CombinedLoss()
cbfs = [
    partial(AvgStatsCallback, [combined_accuracy, class_accuracy, seq_accuracy]), 
    partial(CudaCallback, device),
]
learn = Learner(model, data, loss_func, cb_funcs=cbfs)
learn.cbs.pop(0) # Remove `TrainEvalCallback`, which forces model to go in training mode at beginning of epoch
learn.model.eval()
learn.in_train = False
learn.fit(1)
```

And that's it for testing.

To load unlabelled data for pure prediction:

```
text_data = TabularDataset \
    .from_csv('data/text.txt', names=['text']) \
    .label(proc_x=proc_x)
```

Here no need to split or create a data bunch. Instead, we will create our loader directly.

```
from DeeperNlu.data import pad_collate
from torch.utils.data import DataLoader

dl = DataLoader(text_data, batch_size=64, collate_fn=pad_collate))
```

Model is loaded the same way as for testing. Moreover, usually during inference we are interested in saving the predictions in a human-readable format, i.e. we need to prepare to deprocess the model predictions. We also ensure to call the model with `apply_softmax` set to true, because we want to get probabilities for each class.

```
from DeeperNlu.util import compose, listify

deproc_y = {k:[p.deprocess for p in reversed(listify(proc))] for proc in proc_y.items()}

# Somehow get `xb`, whether a manually-constructed tensor, or from a data loader like the one we created above
for xb in dl:
    pred = model(xb, apply_softmax=True)
    intent, labels = pred # let's assume we are working with an NLU model here
    # Grab the top probabilities for each sample of our batch
    intent_probas, intent_indices = intent.max(-1)
    labels_probas, labels_indices = labels.max(-1)
    # Deprocess the output
    intent = compose(intent_indices.tolist(), deproc_y['intent'])
    labels = compose(labels_indices.tolist(), deproc_y['labels'])
    # save the human-readable predictions
```

## Model zoo

DeeperNlu is built on top of PyTorch. We implemented several models and useful model components to be found under the `DeeperNlu.nn` namespace.

Examples of useful components are: a `TimeDistributed` module whose constructor takes a module and thereby allows that module to be applied to a tensor with a time dimension, applying the original module to each timestep. `RecurrentNet` allows you to build a stack of RNNs with or without residual connections between layers. Type of RNNs supported: regular RNN, LSTM, GRU. Attention modules can be found under `DeeperNlu.nn.attention`. We have a CNN encoder for sequences under `DeeperNlu.nn.SequenceEncoder`.

In terms of full models, we have two mutltitask NLU models (predicting both intent and entities): one based on RNNs, and one based on transformers (`DeeperNlu.nn.BertForIcAndNer`).

The model zoo is in active development.
