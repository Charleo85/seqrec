from torch import nn
from transformer import TransformerEncoderLayer, TransformerEncoder
from attention import MultiheadAttention
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math, copy

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# self-attention model with mask
class SATT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, num_heads=1, use_cuda=True, batch_size=50, dropout_input=0, dropout_hidden=0.5, embedding_dim=-1, position_embedding=False, shared_embedding=True, window_size=8, kernel_type='exp-1', contextualize_opt=None):
        super().__init__()
        
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        
        self.embed = nn.Embedding(input_size, hidden_size, padding_idx=0).to(self.device)
        self.pe = PositionalEncoding(hidden_size, dropout_input,  max_len=window_size)

        if shared_embedding:
            self.out_matrix = self.embed.weight.to(self.device)
        else:
            self.out_matrix = nn.Parameter(torch.rand(output_size, hidden_size, requires_grad=True, device=self.device))

#         self.shared_key = torch.rand(1, batch_size, hidden_size, requires_grad=True).to(self.device)
                
        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward=2048, dropout=dropout_hidden)
        norm = nn.LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, norm=norm).to(self.device)
        
        self.decoder = MultiheadAttention(hidden_size, num_heads, dropout=dropout_hidden)
        
        parts = kernel_type.split('-')
        kernel_types = []

        self.params = []
        for i in range( len(parts) ):
            pi = parts[i]
            if pi in {'exp', 'exp*', 'log', 'lin', 'exp^', 'exp*^', 'log^', 'lin^', 'ind', 'const', 'thres'}:
                if pi.endswith('^'):
                    var = (nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)*5+10 ), nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)))
                    kernel_types.append( pi[:-1] )
                else:
                    var = (nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)*0.01), nn.Parameter(torch.rand(1, requires_grad=True, device=self.device) )  )
                    kernel_types.append( pi )
                    
                self.register_parameter(pi+str(len(self.params))+'_0',  var[0] )
                self.register_parameter(pi+str(len(self.params))+'_1',  var[1] )
                
                self.params.append( var )
                
            elif pi.isdigit():
                val = int(pi)
                if val > 1:
                    pi = parts[i-1]
                    for j in range(val-1):
                        if pi.endswith('^'):
                            var = (nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)*5+10 ), nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)))
                            kernel_types.append( pi[:-1] )
                        else:
                            var = (nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)*0.01), nn.Parameter(torch.rand(1, requires_grad=True, device=self.device) )  )
                            kernel_types.append( pi )
                            
                        
                        self.register_parameter(pi+str(len(self.params))+'_0',  var[0] )
                        self.register_parameter(pi+str(len(self.params))+'_1',  var[1] )
                        
                        self.params.append( var )

            else:
                print('no matching kernel '+ pi) 
                
        self.kernel_num = len(kernel_types)
        print(kernel_types, self.params)
            
        def decay_constructor(t):
            kernels = []
            for i in range( self.kernel_num ):
                pi = kernel_types[i]
                if pi == 'log':
                    kernels.append( torch.mul( self.params[i][0] , torch.log1p(t) ) + self.params[i][1] )
                elif pi == 'exp':
                    kernels.append(  1000* torch.exp( torch.mul( self.params[i][0], torch.neg( t ) ) ) + self.params[i][1] )
                elif pi == 'exp*':
                    kernels.append(  torch.mul( self.params[i][0], torch.exp( torch.neg( t ) ) ) + self.params[i][1] )
                elif pi == 'lin':
                    kernels.append( self.params[i][0] * t  + self.params[i][1] )
                elif pi == 'ind':
                    kernels.append( t )
                elif pi == 'const':
                    kernels.append( torch.ones(t.size(), device=self.device ) )
                elif pi == 'thres':
                    kernels.append( torch.reciprocal( 1 + torch.exp( -self.params[i][0] * t + self.params[i][1] ) )  )
                    
            return torch.stack( kernels, dim=2)
                
        self.decay = decay_constructor   
            
        self.contextualize_opt = contextualize_opt
        if self.contextualize_opt == 'item_subspace':
            subspace_size = 10
            bidirectional = True
            self.gru = nn.GRU(hidden_size, subspace_size, num_layers=1, dropout=dropout_hidden, batch_first=True, bidirectional=bidirectional)
            self.gru2context = nn.Linear( 20 , self.kernel_num)

        
        self.hidden_size = hidden_size
        self.batch_size = batch_size
    
        self = self.to(self.device)

    def forward(self, src, t, features, debug=False, target=None):
        x_embed = self.embed(src)
        src_mask = (src == 0)
        src_mask_neg = (src != 0)
        
        x = x_embed.transpose(0,1)
      
        if self.pe != None:
            x = self.pe(x)  

        if debug:
            x, alpha = self.encoder(x, src_key_padding_mask=src_mask, debug=True )
        else:
            x = self.encoder(x, src_key_padding_mask=src_mask )
            
#         d_output = x[-1,:,:].unsqueeze(0) ### last hidden state 
        
        trg = self.embed(src[:, -1]).unsqueeze(0)  ### last input
        #trg = x[-1, :, :].unsqueeze(0) ### last hidden as key vector
        _, weight = self.decoder(trg, x, x, src_mask, softmax=True)
        alpha = (weight.squeeze(1) * src_mask_neg.float()).unsqueeze(2) 


        t_decay = self.decay(t)
        beta = alpha * t_decay


        if self.contextualize_opt == 'item_subspace':
            output, hidden = self.gru( (x_embed) )
            context = self.gru2context( output )
            context =  F.softmax( context, dim=-1 )
            if debug: print('context weight', context)
            gamma = beta * context
        else: gamma = beta 

        gamma = F.softmax( gamma.masked_fill(src_mask.unsqueeze(2), float('-inf')) , dim=1 )
        gamma = torch.sum( gamma , dim=-1, keepdim=True) 

        x_seq = torch.mul(gamma, x_embed )
        d_output = torch.sum( x_seq , dim=1)        
        
        if debug:
            weight = torch.bmm(x.transpose(0, 1), d_output.squeeze(1).unsqueeze(-1))
            alpha = (weight.squeeze(-1) * src_mask_neg.float()).unsqueeze(2)
            alpha = alpha.masked_fill(src_mask.unsqueeze(2), float('-inf'))
            alpha = F.softmax(alpha, dim=1 )

        output = F.linear(d_output.squeeze(0), self.out_matrix)
        
        if debug:      
            return output, alpha, beta, context, gamma

        return output
