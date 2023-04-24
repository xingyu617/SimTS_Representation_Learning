import torch 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from utils import centerize_vary_length_series, torch_pad_nan,overlap,pad_nan_to_target
import math
import random
from models.augmentation import *
from models.dilation import *
from models.encoder import *
from models.loss import *


# helper functions
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)


class PretrainDataset(Dataset):
    def __init__(self,
                 data,
                 sigma,
                 p=0.5,
                 multiplier=10):
        super().__init__()
        self.data = data
        self.p = p
        self.sigma = sigma
        self.multiplier = multiplier
        self.N, self.T, self.D = data.shape # num_ts, time, dim

    def __getitem__(self, item):
        ts = self.data[item % self.N]
        return self.transform(ts), self.transform(ts)

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma)


def segment(x,overlap):
    B,T,num_elem = x.shape[0],x.shape[1],x.shape[2]
    block = int(T//B)
    index = []
    for i in range(B):
        index.append(range( i*block,(i+1)*block))
    index = torch.tensor(index)
    x = x[torch.arange(index.shape[0])[:,None], index]

    return x

   
class LinearPred(torch.nn.Module):
    def __init__(self,input_dims,input_len,output_dims,output_len):
        super(LinearPred, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
        else:
            self.device = 'cpu'
        self.Wk_wl = nn.Linear(input_len, output_len).to(self.device)
        self.Wk_wl2 = nn.Linear(input_dims, output_dims).to(self.device)
        self.dropout = nn.Dropout(p=0.25)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x_pred = self.relu(self.Wk_wl(x)).transpose(1,2)
        x_pred2 = self.Wk_wl2(x_pred)
       
        return x_pred2


class SimTS:
    '''The SimTS model'''
    
    def __init__(
        self,
        input_dims,
        raw_length,
        K,
        kernel_list: List[int],
        experiment = None,
        experiment_args = {},
        output_dims=320,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        hierarchical_loss: bool = False,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        mix = False,
        
    ):
        ''' Initialize a SimTS model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            K (int): The length of history segmentation.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            hierarchical_loss (bool): Whether to use hierarchical loss.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, NoneType]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function that would be called after each epoch.
        '''
        
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.experiment = experiment
        self.experiment_args=experiment_args
        self.K = K
        self.net = CausalCNNEncoder(in_channels = input_dims, 
                                    reduced_size=320, 
                                    component_dims = output_dims, 
                                    kernel_list= kernel_list).to(self.device)
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        
        self.mix = mix
        
        self.timestep = max_train_length-K
        
        self.dropout = torch.nn.Dropout(p=0.9, inplace=False)
        self.predictor =LinearPred(output_dims,1,output_dims, self.timestep)

        if hierarchical_loss == True:
            self.loss = hierarchical_cosine_loss
        else:
            self.loss = cosine_loss

    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the SimTS model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3

        if n_iters is None and n_epochs is None:
            n_iters = 300 if train_data.size <= 100000 else 600  # default param for n_iters
        
        arrs=[]
        if self.max_train_length is not None:
            index = [i*self.K for i in range(train_data.shape[1] // 201)]
            sections = train_data.shape[1] // self.max_train_length
            if sections!=0:
                segment = train_data.shape[1]//sections
              
                for i in range(len(index)):
                    if index[i]+segment < train_data.shape[1]:
                        arrs.append(train_data[:,index[i]:index[i]+segment,:])
                    else: 
                        arrs.append(train_data[:,-segment:,:])
                    arrs[i] = pad_nan_to_target(arrs[i], arrs[0].shape[1], axis=1)
                train_data = np.concatenate(arrs, axis=0)

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
                
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        # multiplier = 1 if train_data.shape[0] >= self.batch_size else math.ceil(self.batch_size / train_data.shape[0])

        # PretrainDataset to transform the time series data with jittering, scaling, and shifting (https://openreview.net/forum?id=PilZY3omXV2).
        # from https://github.com/salesforce/CoST/blob/afc26aa0239470f522135f470861a1c375507e84/cost.py#L17
        # train_dataset = PretrainDataset(torch.from_numpy(train_data).to(torch.float), sigma=0.5, multiplier=multiplier)
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)

        optimizer = torch.optim.SGD([
            {'params': list(self.net.parameters())},
            {'params': list(self.predictor.parameters()), 'lr': 0.0001*self.lr}
        ], lr=self.lr)

        loss_log = []
        early_stop_step=0
        best_loss=100.0
        para = None
        best_net = None
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            # adjust_learning_rate(optimizer, self.n_epochs, self.lr, n_epochs)
            cum_loss = 0
            n_epoch_iters = 0

            interrupted = False
            for seq in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                if self.max_train_length is not None and seq[0].size(1) > self.max_train_length:
                    window_offset = np.random.randint(seq[0].size(1) - self.max_train_length + 1)
                    x = seq[0][:, window_offset : window_offset + self.max_train_length]
                else:
                    x = seq[0]
                    
                x1 = x[:,:self.K,:].clone().to(self.device)
                x2 = x[:,self.K:self.max_train_length,:].clone().to(self.device)
               
                optimizer.zero_grad()
                
                torch.cuda.empty_cache()
                
                z1, _, z2= self.net(x1, x2, mask = None)
                
                rand_idx = random.randint(127, z1.shape[1]-1)
                # trend_h_repr = lasts
                trend1 = z1[:,rand_idx,:]
                # print(rand_idx)
                
                # TODO
                encode_future_embeds = z2.to(self.device) # no-gradient
                fcst_future_embeds = self.predictor(trend1.unsqueeze(-1))

                loss = self.loss(encode_future_embeds, fcst_future_embeds)

                loss.backward()
                optimizer.step()
                # self.net.update_parameters(self._net)
                # self.net = self._net
                cum_loss += loss.item()
                n_epoch_iters += 1

                self.n_iters += 1

                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())

            if interrupted:
                break
            
            old_para = para
            para = self.net.print_para()
            if old_para !=None:
              print(torch.equal(para.data, old_para.data))
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if best_loss-cum_loss<=0.0001:
                early_stop_step+=1
            else:
                early_stop_step=0
            if best_loss>cum_loss:
                best_loss=cum_loss
                best_net = self.net
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1

            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log,best_net
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        
        _, out, _= self.net(x.to(self.device, non_blocking=True), train = False)
       
        out = out.unsqueeze(1)
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        if data.ndim !=3:
            data=data.unsqueeze(0)
            
        assert data.ndim == 3
        
        if not isinstance(data,np.ndarray):
            data=data.clone().detach().cpu().numpy()
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        state_dict = torch.load(fn, map_location=self.device)
        self.net.load_state_dict(state_dict)
    

def adjust_learning_rate(optimizer, epoch, lr, total_epochs):
    import math
    """Decay the learning rate based on schedule"""
    for param_group in optimizer.param_groups:
        param_group["lr"] *= 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))