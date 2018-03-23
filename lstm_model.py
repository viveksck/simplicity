from seqtools import * 
import configargparse
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
torch.manual_seed(1)
from torch.autograd import Variable
import numpy as np
import sys
# Minimum count
MIN_COUNT = 1
# Min length of input and output sequence
MIN_LENGTH = 1
# Max length of input and output sequence
MAX_LENGTH = 30


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size=1, dropout=0.0, n_layers=1):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers=n_layers
        global use_pretrained, pre_trained_embeddings
        if use_pretrained:
            self.word_embeddings = nn.Embedding(pre_trained_embeddings.size(0), pre_trained_embeddings.size(1))
            self.word_embeddings.weight = nn.Parameter(pre_trained_embeddings)
            self.word_embeddings.weight.requires_grad = False
            print "Using pretrained embeddings"
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout, num_layers=n_layers)


        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(batch_size)
        self.tagset_size = tagset_size

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.batch_size=batch_size
        if USE_CUDA:
            return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim).cuda()),
                    autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)).cuda())
        else:
            return (autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)),
                    autograd.Variable(torch.zeros(self.n_layers, batch_size, self.hidden_dim)))

    def forward(self, sentences):
        embeds = self.word_embeddings(sentences)
        ems=embeds.view(embeds.size()[0], embeds.size()[1], embeds.size()[2])
        lstm_outputs, self.hidden = self.lstm(ems, self.hidden)
        lstm_output_reshaped = lstm_outputs.transpose(0,1).contiguous() # Transpose to have batch first
        tag_space= self.hidden2tag(lstm_output_reshaped.view(-1, lstm_outputs.size()[2])) # concatenate all batches
        return tag_space.view(self.batch_size, -1, self.tagset_size)
        #logits = F.softmax(tag_space).view(self.batch_size, -1, self.tagset_size)
        #return logits

def filter_pairs(pairs):
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs

def get_validation_batch(pairs, input_lang, output_lang):                                                                                                                        
    input_seqs = []                                                                                                                                                              
    target_seqs = []                                                                                                                                                             
                                                                                                                                                                                 
    # Choose random pairs                                                                                                                                                        
    for pair in pairs:                                                                                                                                                          
        input_seqs.append(indexes_from_sentence(input_lang, pair[0]))                                                                                                            
        target_seqs.append(indexes_from_sentence(output_lang, pair[1]))                                                                                                          
                                                                                                                                                                                 
    # Zip into pairs, sort by length (descending), unzip                                                                                                                         
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)                                                                                      
    input_seqs, target_seqs = zip(*seq_pairs)                                                                                                                                   
                                                                                                                                                                                 
    # For input and target sequences, get array of lengths and pad with 0s to max length                                                                                         
    input_lengths = [len(s) for s in input_seqs]                                                                                                                                 
    input_padded = [pad_seq(s, MAX_LENGTH) for s in input_seqs]                                                                                                          
    target_lengths = [len(s) for s in target_seqs]                                                                                                                               
    target_padded = [pad_seq(s, MAX_LENGTH) for s in target_seqs]                                                                                                       
                                                                                                                                                                                 
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)                                                                              
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)                                                                                                         
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)                                                                                                       
                                                                                                                                                                                 
    if USE_CUDA:
	input_var = input_var.cuda()                                                                                                                                             
        target_var = target_var.cuda()                                                                                                                                           
                                                                                                                                                                                 
    return input_var, input_lengths, target_var, target_lengths                                                                                                                  

def get_validation_loss(validation_pairs, input_lang, output_lang, lstm_model, batch_size=1):
    num_batches = int(np.ceil(len(validation_pairs)/float(batch_size)))
    val_losses_batches=[]
    for chunk in more_itertools.chunked(validation_pairs, batch_size):
        input_var, input_lengths, target_var, target_lengths = get_validation_batch(chunk, input_lang, output_lang)
        val_loss_batch =evaluate_batch(lstm_model, input_var,  target_var, target_lengths, batch_size)
        val_losses_batches.append(val_loss_batch)
    return np.mean(val_losses_batches)
    
# Building the models


def evaluate_batch(lstm_model, input_batches, target_batches, target_lengths, batch_size):
    lstm_model.zero_grad()
    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    lstm_model.hidden = lstm_model.init_hidden(batch_size=batch_size)
    tag_scores = lstm_model(input_batches)
    tag_scores=tag_scores.contiguous()
    target_batches=target_batches.transpose(0,1).contiguous()
    #print "Target batches", target_batches.size()
    loss=masked_cross_entropy(tag_scores, target_batches, target_lengths)
    return loss.data[0]

# In[23]:
def train(input_batches, input_lengths, target_batches, target_lengths, lstm_model, lstm_model_scheduler, batch_size, clip, max_length=MAX_LENGTH, validation_pairs=None, input_lang=None, output_lang=None, epoch_no=-1):
    
    assert(validation_pairs is not None)
    assert(epoch_no != -1)
    # Zero gradients of both optimizers
    lstm_model_scheduler.optimizer.zero_grad()
    lstm_model.zero_grad()
    loss = 0.0 # Added onto for each word
    # Also, we need to clear out the hidden state of the LSTM,
    # detaching it from its history on the last instance.
    lstm_model.hidden = lstm_model.init_hidden(batch_size=batch_size)
    tag_scores = lstm_model(input_batches)
    tag_scores=tag_scores.contiguous()
    target_batches=target_batches.transpose(0,1).contiguous()
    loss=masked_cross_entropy(tag_scores, target_batches, target_lengths)
    loss.backward()
    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(lstm_model.parameters(), clip)
    lstm_model_scheduler.optimizer.step()
    val_loss = get_validation_loss(validation_pairs, input_lang, output_lang, lstm_model)
    # Update parameters with optimizers
    lstm_model_scheduler.step(metrics=val_loss, epoch=epoch_no)
    return loss.data[0], ec, val_loss

# ## Running training
# 
# With everything in place we can actually initialize a network and start training.
# 
# To start, we initialize models, optimizers, a loss function (criterion), and set up variables for plotting and tracking progress:

# In[24]:


# Configure models
def configure_and_train(input_lang, output_lang, pairs, validation_pairs, lstm_model_params, mlconfig):
    embedding_dim = mlconfig['embedding_dim']
    hidden_size = mlconfig['hidden_size']
    n_layers = mlconfig['n_layers']
    dropout = mlconfig['dropout']
    batch_size = mlconfig['batch_size']
    # Configure training/optimization
    clip = mlconfig['clip']
    learning_rate = mlconfig['learning_rate']
    n_epochs = mlconfig['n_epochs']
    plot_every = mlconfig['plot_every']
    print_every = mlconfig['print_every']
    evaluate_every = mlconfig['evaluate_every']
    save_every = mlconfig['save_every']
    patience = mlconfig['patience']
    cooldown = mlconfig['cooldown']

    epoch = 0
    # Initialize models
    lstm_model = LSTMTagger(embedding_dim, hidden_size, input_lang.n_words, output_lang.n_words, batch_size, dropout, n_layers=n_layers)
    # Initialize optimizers and criterion
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, lstm_model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
       lstm_model.cuda()

    # Keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    plot_losses_val = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    print_loss_total_val = 0
    plot_loss_total_val = 0

    # Begin!
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    lstm_model_scheduler = EarlyStopper(optimizer, 'min', factor=0.0, verbose=True, patience=patience, cooldown=cooldown)
    best_val_loss=10.0

    while epoch < n_epochs:
        epoch += 1
    	# Get training data for this cycle
    	input_batches, input_lengths, target_batches, target_lengths = random_batch(input_lang, output_lang, pairs, batch_size)
    	# Run the train function
    	loss, ec, val_loss = train(
        	input_batches, input_lengths, target_batches, target_lengths,
        	lstm_model,
        	lstm_model_scheduler,
                batch_size, clip,
                MAX_LENGTH, validation_pairs, input_lang, output_lang, epoch)
    	# Keep track of loss
    	print_loss_total += loss
    	plot_loss_total += loss
        print_loss_total_val+=val_loss
        plot_loss_total_val+=val_loss

    	eca += ec

        if val_loss < best_val_loss:
            torch.save(lstm_model.state_dict(), lstm_model_params + "_best")
            best_val_loss = val_loss

    	if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_loss_avg_val = print_loss_total_val / print_every
            print_loss_total_val = 0
            print_summary = '%s (%f %f%%) training: %.4f val: %.4f' % (time_since(start, epoch / float(n_epochs)), epoch, epoch / float(n_epochs) * 100, print_loss_avg, print_loss_avg_val)
            print(print_summary)
        
    	if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_avg_val = plot_loss_total_val / plot_every
            plot_losses.append(plot_loss_avg)
            plot_losses_val.append(plot_loss_avg_val)
            plot_loss_total = 0
            plot_loss_total_val = 0
        
            # TODO: Running average helper
            ecs.append(eca / plot_every)
            ecs_win = 'lstm_model grad (%s)' % hostname + input_lang.name + "-" +  output_lang.name
            tcs_win = 'training loss (%s)' % hostname + input_lang.name + "-" + output_lang.name
            vcs_win = 'validation loss (%s)' % hostname + input_lang.name + "-" + output_lang.name
            vis.line(np.array(ecs), win=ecs_win, opts={'title': ecs_win})
            vis.line(np.array(plot_losses), plot_every*np.arange(1, len(np.array(plot_losses))+1), win=tcs_win, opts={'title': tcs_win})
            vis.line(np.array(plot_losses_val), plot_every*np.arange(1, len(np.array(plot_losses_val))+1), win=vcs_win, opts={'title': vcs_win})
            eca = 0
            dca = 0

        if (epoch % save_every) == 0:
            torch.save(lstm_model.state_dict(), lstm_model_params + "_{}".format(epoch))

        if lstm_model_scheduler.stop_status:
            print "Stopping early"
            break

    torch.save(lstm_model.state_dict(), lstm_model_params + "_{}".format(epoch))


def read_langs(lang1, lang2, lang1_word_tokenizer, lang1_word_joiner, lang2_word_tokenizer, lang2_word_joiner, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    filename = './data/%s-%s.csv' % (lang1, lang2)
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    source_lines=[s.strip() for s in df.source.values]
    target_lines=[t.strip() for t in df.target.values]
    pairs=zip(source_lines, target_lines)
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2, lang2_word_tokenizer, lang2_word_joiner)
        output_lang = Lang(lang1, lang1_word_tokenizer, lang1_word_joiner)
    else:
        input_lang = Lang(lang1, lang1_word_tokenizer, lang1_word_joiner)
        output_lang = Lang(lang2, lang2_word_tokenizer, lang2_word_joiner)

    print("Filtered to %d pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
    print "Trimming languages"
    input_lang.trim(MIN_COUNT)
    output_lang.trim(MIN_COUNT)
    return input_lang, output_lang, pairs

def prepare_data(lang1_name, lang2_name, lang1_word_tokenizer, lang1_word_joiner, lang2_word_tokenizer, lang2_word_joiner, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, lang1_word_tokenizer, lang1_word_joiner, lang2_word_tokenizer, lang2_word_joiner, reverse)
    print("Read %d sentence pairs" % len(pairs))
    keep_pairs = []
    for pair in pairs:
    	input_sentence = pair[0]
    	output_sentence = pair[1]
    	keep_input = True
    	keep_output = True
    	for word in input_lang.word_tokenizer(input_sentence):
            if word and word not in input_lang.word2index:
                keep_input = False
            	break

    	for word in output_lang.word_tokenizer(output_sentence):
            if word and word not in output_lang.word2index:
            	keep_output = False
                break

    	# Remove if pair doesn't match input and output conditions
    	if keep_input and keep_output:
        	keep_pairs.append(pair)

    print("Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / float(len(pairs))))
    pairs = keep_pairs

    print "Testing we can get a random batch."
    random_batch(input_lang, output_lang, pairs, 2)

    print "Testing models"

    return input_lang, output_lang, pairs

def main(args):
    global use_pretrained, pre_trained_embeddings 

    source_lang = args.source_lang
    target_lang = args.target_lang
    lstm_model_params = args.lstm_model_params
    mlconfig={}
    mlconfig['embedding_dim'] = args.embedding_dim
    mlconfig['hidden_size'] = args.hidden_size
    mlconfig['n_layers'] = args.n_layers
    mlconfig['dropout'] = args.dropout
    mlconfig['batch_size'] = args.batch_size
    # Configure training/optimization 
    mlconfig['clip'] = args.clip
    mlconfig['learning_rate'] = args.learning_rate
    mlconfig['n_epochs'] = args.n_epochs
    mlconfig['plot_every'] = args.plot_every
    mlconfig['print_every'] = args.print_every
    mlconfig['evaluate_every'] = args.evaluate_every
    mlconfig['save_every'] = args.save_every
    mlconfig['patience'] = args.patience
    mlconfig['cooldown'] = args.cooldown

    input_lang, output_lang, pairs = prepare_data(source_lang, target_lang, split_by_char_tokenizer, join_by_char_tokenizer, split_by_char_tokenizer, join_by_char_tokenizer, False)
    print output_lang.word2index

    if args.embedding_file is not None:
        print "Loading pretrained embeddings"
        emb_model = gensim.models.KeyedVectors.load_word2vec_format(args.embedding_file)
        E=[]
        for i,w in sorted(input_lang.index2word.items()):
            print i, w
            E.append(emb_model[w])
        embeddings=torch.from_numpy(np.array(E))
        use_pretrained = True
        pre_trained_embeddings=embeddings
        print "Created embeddings"

    print "Testing we can get a random batch."
    random_batch(input_lang, output_lang, pairs, 2)

    print "Testing models"
    print "Dumping splits"
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.1, random_state=args.seed)
    train_pairs, validate_pairs = train_test_split(train_pairs, test_size=0.1, random_state=args.seed)
    print "Training data size", len(train_pairs)
    print "Validation data size", len(validate_pairs), validate_pairs[:10]
    print "Test data size", len(test_pairs)
    pickle.dump((input_lang, output_lang, train_pairs, validate_pairs, test_pairs), open(args.file_params, 'wb'))

    print "Training"
    configure_and_train(input_lang, output_lang, train_pairs[:], validate_pairs[:], args.lstm_model_params,  mlconfig)
    return


if __name__ == "__main__":
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='config file path')
    parser.add("-s", "--source_lang", dest="source_lang", help="Source language")
    parser.add("-t", "--target_lang", dest="target_lang", help="Target language")
    parser.add("-e", "--lstm_model_params", dest="lstm_model_params", help="Encoder params")
    parser.add("-o", "--file_params", dest="file_params", help="File to dump the training and validation split")
    parser.add("-m", "--embedding", dest="embedding_file", help="Pretrained word embeddings file")
    parser.add('--embedding_dim', dest='embedding_dim', type=int)
    parser.add('--hidden_size', dest='hidden_size', type=int)
    parser.add('--n_layers', dest='n_layers', type=int)
    parser.add('--dropout', dest='dropout', type=float)
    parser.add('--batch_size', dest='batch_size', type=int)
    parser.add('--clip', dest='clip', type=float)
    parser.add('--learning_rate', dest='learning_rate', type=float)
    parser.add('--n_epochs', dest='n_epochs', type=int)
    parser.add('--plot_every', dest='plot_every', type=int)
    parser.add('--print_every', dest='print_every', type=int)
    parser.add('--evaluate_every', dest='evaluate_every', type=int)
    parser.add('--save_every', dest='save_every', type=int)
    parser.add('--patience', dest='patience', type=float)
    parser.add('--cooldown', dest='cooldown', type=float)
    parser.add('--seed', dest='seed', default=42, type=int)
    args = parser.parse_args()
    main(args)
