import os
from datetime import datetime
import argparse

import numpy as np
import torch

import sys
from src_freeze.wnet import WNet
from src_freeze.crf import crf_batch_fit_predict, crf_fit_predict
sys.path.insert(1, '../../')
from utils.visualise import visualise_outputs
from utils.callbacks import model_checkpoint
import h5py
import skimage.measure
import glob
from pyhdf.SD import SD, SDC


# Read in hdf files
def read_hdf(filename, hdf_key='Data-Set-2'):
    
    hdf = SD(filename, SDC.READ)
    hdf_data = hdf.select(hdf_key)
    hdf_img = hdf_data.get()

    return hdf_img


# Class for correction
class correct():
    
    # Parameters
    def __init__(self):
        # Downsampling factor
        self.bad_pixels = -9999
    
    # Option
    def method_0(self, x):
        return x
    
    # Option
    def method_1(self, x):
        lower_threshold = np.where(x < 0)
        upper_threshold = np.where(x > 3.5)
        x[lower_threshold] = 0
        x[upper_threshold] = 3.5
        return x
    
    # Option
    def method_2(self, x):
        return torch.from_numpy(skimage.measure.block_reduce(x, (self.reduction, self.reduction), np.nanmax))

    # Option: Set bad pixels = minimum value
    def method_3(self, x):
        bad_pixels = np.where(x < -9000)
        not_bad_pixels = np.where(x > -9000)
        x[bad_pixels] = x[not_bad_pixels].min()
        return x

    # Option: Set bad pixels = mean global intensity
    def method_4(self, x):
        bad_pixels = np.where(x < -9000)
        not_bad_pixels = np.where(x > -9000)
        x[bad_pixels] = x[not_bad_pixels].mean()
        return x
        
    # Option: Set bad pixels = global minimum value
    def method_5(self, x):
        lower_threshold = np.where(x < -2.140)
        upper_threshold = np.where(x > 4.149)
        x[lower_threshold] = -2.40
        x[upper_threshold] = 4.149
        return x
        
    # Option: Set bad pixels = -5 (small negative number)
    def method_6(self, x):
        bad_pixels = np.where(x < -9000)
        x[bad_pixels] = -5
        return x

    # Option: Shift entire euv map by +9999
    def method_7(self, x):
        x = x + 9999
        return x
        
    # Option: Raise entire euv map as 10^(euv data)
    def method_8(self, x):
        x = 10**x
        return x
        
    # Option: Raise entire euv map by + global min (-2.140) and set bad pixels to 0
    def method_9(self, x):
        x = x + 2.140
        bad_pixels = np.where(x < -9000)
        x[bad_pixels] = 0
        return x
    
    # Applied rule
    def correct_rule(self, x, case):
        method_name = 'method_' + str(case)
        method = getattr(self, method_name, lambda: 'Invalid Correction Rule')
        return method(x)


# Class for downsampling
class downsampling():
    
    # Parameters
    def __init__(self, reduction):
        
        # Downsampling factor
        self.reduction = reduction
    
    # Option
    def method_0(self, x):
        return x
    
    # Option
    def method_1(self, x):
        return torch.from_numpy(skimage.measure.block_reduce(x, (self.reduction, self.reduction), np.nanmin))
    
    # Option
    def method_2(self, x):
        return torch.from_numpy(skimage.measure.block_reduce(x, (self.reduction, self.reduction), np.nanmax))
    
    # Applied rule
    def downsampling_rule(self, x, case):
        method_name = 'method_' + str(case)
        method = getattr(self, method_name, lambda: 'Invalid Downsampling Rule')
        return method(x)


# Class for normalization
class norm():
    
    # Parameters
    def __init__(self, min_val, max_val, mean_val, stddev_val, median_val):
        
        # Properties
        self.min = min_val
        self.max = max_val
        self.mean = mean_val
        self.stddev = stddev_val
        self.median = median_val

    # Option
    def method_0(self, x):
        return x

    # Option
    def method_1(self, x):
        x = (x - 1) / (self.max - 1)
        return x
    
    # Option
    def method_2(self, x):
        x = (x - self.min) / (self.max - self.min)
        return x
        
    # Option
    def method_3(self, x):
        x = (x - 0) / (3.5 - 0)
        return x
        
    # Option
    def method_4(self, x):
        x = (x - (-2.140)) / (4.149 - (-2.140))
        return x

    # Applied rule
    def norm_rule(self, x, case):
        method_name = 'method_' + str(case)
        method = getattr(self, method_name, lambda: 'Invalid Normalization Rule')
        return method(x)
    
    


# Commands
p = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-action', type=str, default='test',
               help='Start, continue, or test')
p.add_argument('-data_path', type=str, default='../euv/',
               help='path of EUV images')
p.add_argument('-output_path', type=str, default='output/',
               help='directory in which you save wnet predictions')
p.add_argument('-probMap_path', type=str, default='../../neurips_kmeans/outputs_soft_kmeans_withBadPixels/',
               help='path of soft Kmeans probability maps') #benoit
p.add_argument('-model', type=str, default='models/wnet.pt',
               help='path of pre-trained model')
p.add_argument('-epochs', type=int, default=100,
               help='how many epochs to train')
p.add_argument('-reduction', type=int, default=1,
               help='Factor by which to reduce image dimensions')
p.add_argument('-reduction_method', type=int, default=1,
               help='Method used to reduce image dimensions')
p.add_argument('-normalization_method', type=int, default=1,
               help='Method used to normalize input data')
p.add_argument('-correction_method', type=int, default=1,
               help='Method used to correct input data')
p.add_argument('-train_samples', type=int, default=7,
               help='Nb. of samples to use during training process')
p.add_argument('-valid_samples', type=int, default=3,
               help='Nb. of samples to use during validation process')
p.add_argument('-win_size', type=int, default=1,
               help='window size of pre-downsamplling')
p.add_argument('-batch_size', type=int, default=1,
               help='batch size of training')
p.add_argument('-min_epochs', type=int, default=3,
               help='Minimum epochs performed before potential early stopping')               
p.add_argument('-vis_num', type=int, default=1,
               help='number of visualised test images')
p.add_argument('-test_range', nargs='+', type=int,
               default=[2556, 2557],
               help='range of test cases (make sure no missing samples in between)')
p.add_argument('-num_classes', type=int, default=20,
               help='number of classes during clustering')
p.add_argument('-lr', type=float, default=2e-5,
               help='learning rate')
p.add_argument('-per_train', type=float, default=0.7,
               help='percentage samples for training')
p.add_argument('-weight_decay', type=float, default=2e-5,
               help='regularisation factor')
p.add_argument("-cuda", action='store_true',
               help="GPU or CPU")
p.add_argument("-plot_loss", action='store_true',
               help="Plot training and validation loss")
p.add_argument("-test_save", action='store_true',
               help="save outputs for testing")
p.add_argument("-visualise", action='store_true',
               help="if do visualisation")
p.add_argument("-freeze_encoder", type=int, default=0,
               help="1 if you would like to freeze the encoder, 0 if not")
p.add_argument("-freeze_decoder", type=int, default=0,
               help="1 if you would like to freeze the decoder, 0 if not")
args = p.parse_args()

# ----------------------------------------------------------------------------------


print(r'---------------------- LOADING ----------------------')
# Load training & validation data
if args.action == 'start' or args.action == 'continue':
    
    # Linnea: added filenames to include soft_kmeans map names, which are saved with chm filenames
    # Path to training/validation data
    sets = np.load(args.data_path+'/training_validation_test_sets.npz')
    train_filenames = sets['train_euv_filenames']
    train_chm_filenames = sets['train_chm_filenames']
    valid_filenames = sets['valid_euv_filenames']
    valid_chm_filenames = sets['valid_chm_filenames']
    
    # Define training set
    if len(train_filenames) > args.train_samples:
        train_nb = args.train_samples
    else:
        train_nb = len(train_filenames)
    t_i = 0
    t_f = t_i + train_nb
    train_filenames = train_filenames[t_i:t_f]
    
    # Define validation set
    if len(valid_filenames) > args.valid_samples:
        valid_nb = args.valid_samples
    else:
        valid_nb = len(valid_filenames)
    t_i = 0
    t_f = t_i + valid_nb
    valid_filenames = valid_filenames[t_i:t_f]
    
    # Image dimensions
    nx_data = 1600
    ny_data = 3200
    nx = int(nx_data / args.reduction)
    ny = int(ny_data / args.reduction)
    
    # Nb. of channels
    nb_channels = 1
    
    # Linnea: added if statements to load the correct data depending on freeze parameters.
    
    # Linnea: if freeze_decoder or freeze_encoder = 1, only training encoder/decoder. In this case need x_train and x_valid to be images
    # and need y_train and y_valid probability maps to compare to.

    # benoit : We'll keep the same order what x and y represent
    if (args.freeze_decoder == 1 or args.freeze_encoder == 1):
        # Arrays/tensors
        x_train = torch.zeros([train_nb, nb_channels, nx, ny])
        x_valid = torch.zeros([valid_nb, nb_channels, nx, ny])
        y_train = torch.zeros([train_nb, args.num_classes, nx, ny])
        y_valid = torch.zeros([valid_nb, args.num_classes, nx, ny])
        
        # Linnea: read in images for x_train and x_valid
        # Training set
        downsample = downsampling(args.reduction)
        for i in range(train_nb):
            print('Training set {0}/{1}'.format(i+1, train_nb, train_filenames[i]))
            if args.reduction > 1:
                x = read_hdf(args.data_path+'/'+train_filenames[i])
                x_train[i, 0, :, :] = downsample.downsampling_rule(x, args.reduction_method)
            else:
                x_train[i, 0, :, :] = read_hdf(train_filenames[i])
    
        # Correct
        corr = correct()
        x_train = corr.correct_rule(x_train, args.correction_method)
        
        # Validation set
        for i in range(valid_nb):
            print('Validation set {0}/{1}, {2}'.format(i+1, valid_nb, valid_filenames[i]))
            if args.reduction > 1:
                x = read_hdf(args.data_path+'/'+valid_filenames[i])
                x_valid[i, 0, :, :] = downsample.downsampling_rule(x, args.reduction_method)
            else:
                x_valid[i, 0, :, :] = read_hdf(valid_filenames[i])
        
        # Correct
        x_valid = corr.correct_rule(x_valid, args.correction_method)
        
        # Normalization
        norm_train = norm(x_train.min(), x_train.max(), x_train.mean(), x_train.std(), x_train.median())
        x_train = norm_train.norm_rule(x_train, args.normalization_method)
        norm_valid = norm(x_valid.min(), x_valid.max(), x_valid.mean(), x_valid.std(), x_valid.median())
        x_valid = norm_valid.norm_rule(x_valid, args.normalization_method)
        
        # Linnea: read in probability maps for y_train and y_valid
        # Read in y_train
        for i in range(train_nb):
            filename = args.probMap_path + train_chm_filenames[i] + '5'
            print('Training set {0}/{1}'.format(i+1, train_nb, train_chm_filenames[i]))
            # print('Training input: {0}'.format(filename))
            hf = h5py.File(filename, 'r')
            # Modified to switch order of indices
            data = np.transpose(hf.get('kmeans_clusters'), (2, 0, 1))  # [()] #.value
            y_train[i, :, :, :] = torch.tensor(data)
            
        # Read in y_valid
        for i in range(valid_nb):
            filename = args.probMap_path + valid_chm_filenames[i] + '5'
            print('Validation set {0}/{1}, {2}'.format(i+1, valid_nb, valid_chm_filenames[i]))
            # print('Validation input: {0}'.format(filename))
            hf = h5py.File(filename, 'r')
            data = np.transpose(hf.get('kmeans_clusters'), (2, 0, 1))  # [()] #.value
            y_valid[i, :, :, :] = torch.tensor(data)
        
        # Linnea: There are now more shapes to check (x & y)
        # Shapes
        print('Shape of x_train:', x_train.shape)
        print('Shape of x_valid:', x_valid.shape)
        print('Shape of y_train:', y_train.shape)
        print('Shape of y_valid:', y_valid.shape)

    # Linnea: if nothing is frozen, should run as usual (only x_train and x_valid, and they both equal y_train and y_valid)
    if (args.freeze_encoder == 0 and args.freeze_decoder == 0):
        # Arrays/tensors
        x_train = torch.zeros([train_nb, nb_channels, nx, ny])
        x_valid = torch.zeros([valid_nb, nb_channels, nx, ny])
     
        # Linnea: read in images for x_train and x_valid
        # Training set
        downsample = downsampling(args.reduction)
        for i in range(train_nb):
            print('Training set {0}/{1}'.format(i+1, train_nb))
            if args.reduction > 1:
                x = read_hdf(args.data_path+'/'+train_filenames[i])
                x_train[i, 0, :, :] = downsample.downsampling_rule(x, args.reduction_method)
            else:
                x_train[i, 0, :, :] = read_hdf(train_filenames[i])
    
        # Correct
        corr = correct()
        x_train = corr.correct_rule(x_train, args.correction_method)
        
        # Validation set
        for i in range(valid_nb):
            print('Validation set {0}/{1}, {2}'.format(i+1, valid_nb, valid_filenames[i]))
            if args.reduction > 1:
                x = read_hdf(args.data_path+'/'+valid_filenames[i])
                x_valid[i, 0, :, :] = downsample.downsampling_rule(x, args.reduction_method)
            else:
                x_valid[i, 0, :, :] = read_hdf(valid_filenames[i])
        
        # Correct
        x_valid = corr.correct_rule(x_valid, args.correction_method)
        
        # Linnea: debugging
        print("Pre-normalization statistics of x_train (min, max, mean, std, median): ", x_train.min(), x_train.max(), x_train.mean(), x_train.std(), x_train.median())
        print("Pre-normalization statistics of x_valid (min, max, mean, std, median): ", x_valid.min(), x_valid.max(), x_valid.mean(), x_valid.std(), x_valid.median())
        
        # Normalization
        norm_train = norm(x_train.min(), x_train.max(), x_train.mean(), x_train.std(), x_train.median())
        x_train = norm_train.norm_rule(x_train, args.normalization_method)
        norm_valid = norm(x_valid.min(), x_valid.max(), x_valid.mean(), x_valid.std(), x_valid.median())
        x_valid = norm_valid.norm_rule(x_valid, args.normalization_method)
        
        # Linnea: debugging
        print("Post-normalization statistics of x_train (min, max, mean, std, median): ", x_train.min(), x_train.max(), x_train.mean(), x_train.std(), x_train.median())
        print("Post-normalization statistics of x_valid (min, max, mean, std, median): ", x_valid.min(), x_valid.max(), x_valid.mean(), x_valid.std(), x_valid.median())
        
        # Linnea: add y_train and y_valid which are duplicates of x_train and x_valid to make later code simple for 
        # all freezing scenarios.
        y_train = x_train
        y_valid = x_valid
        
        # Linnea: There are now more shapes to check (x & y)
        # Shapes
        print('Shape of x_train:', x_train.shape)
        print('Shape of x_valid:', x_valid.shape)
        print('Shape of y_train:', y_train.shape)
        print('Shape of y_valid:', y_valid.shape)


if args.action == 'start' or args.action == 'continue':

    #benoit
    print(r'---------------------- TRAINING ----------------------')
    if args.action == 'continue':
        net = torch.load(args.model)
    else:
        net = WNet(num_channels=x_train.shape[1],
                   num_classes=args.num_classes)
                   
    if args.freeze_encoder == 1:
        for param in net.encoder.parameters():
                param.requires_grad = False
    else:
        for param in net.encoder.parameters():
                param.requires_grad = True
                
    if args.freeze_decoder == 1:
        for param in net.decoder.parameters():
                param.requires_grad = False
    else:
        for param in net.decoder.parameters():
                param.requires_grad = True

    if args.cuda:
        net = net.cuda()

    # if args.action == 'train':
    #     date = datetime.now().__str__()
    #     date = date[:16].replace(':', '-').replace(' ', '-')

    # Linnea: Added y_train and y_valid to net.fit
    # See src_freeze/network.py and src_freeze/wnet.py
    print('Beginning training: ', args.freeze_encoder, args.freeze_decoder)
    net.fit(
        x_train, y_train,
        x_valid, y_valid,
        epochs=args.epochs,
        learn_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        callbacks=[
            # model_checkpoint(os.path.join('models', f'wnet-{date}.pt'))
            model_checkpoint(args.model)
        ],
        plot=args.plot_loss,
        min_epochs = args.min_epochs,
        freeze_decoder = args.freeze_decoder,
        freeze_encoder = args.freeze_encoder
    )
    torch.cuda.empty_cache()



elif args.action == 'test':
    
    print(r'---------------------- TESTING ----------------------')

    # Weights
    if args.model:
        net = torch.load(args.model)
        
    # Cuda
    if args.cuda:
        net = net.cuda()

    # Image dimensions
    nx_data = 1600
    ny_data = 3200
    nx = int(nx_data / args.reduction)
    ny = int(ny_data / args.reduction)
    
    # Nb. of channels
    nb_channels = 1
    
    # Nb. of classes
    nb_classes = args.num_classes
    
    # Path to test data
    sets = np.load(args.data_path+'/training_validation_test_sets.npz')
    test_filenames = sets['test_euv_filenames'][args.test_range[0]:args.test_range[1]]
    output_filenames = sets['test_chm_filenames'][args.test_range[0]:args.test_range[1]]
    test_nb = len(test_filenames)
    # Nb. of test performed at a time
    nb_tests = 1

    # Arrays/tensors
    x_test = torch.zeros([nb_tests, nb_channels, nx, ny])
    
    for i in range(test_nb):
        print('Test {0}'.format(test_filenames[i]))
        
        # Image dimensions
        downsample = downsampling(args.reduction)
        if args.reduction > 1:
            x = read_hdf(args.data_path+'/'+test_filenames[i])
            x_test[0, 0, :, :] = downsample.downsampling_rule(x, args.reduction_method)
        else:
            x_test[0, 0, :, :] = read_hdf(args.data_path+'/'+test_filenames[i])

        # Correction:
        corr = correct()
        x_test = corr.correct_rule(x_test, args.correction_method)
        
        # Normalization:
        norm_test = norm(x_test.min(), x_test.max(), x_test.mean(), x_test.std(), x_test.median())
        x_test = norm_test.norm_rule(x_test, args.normalization_method)
        
        # Linnea: Added call to read in probability maps for y_test if freeze_encoder or freeze_decoder = 1
        if (args.freeze_decoder == 1 or args.freeze_encoder == 1):
            # Arrays/tensors
            y_test = torch.zeros([nb_tests, args.num_classes, nx, ny])
            
            # Linnea: read in probability maps for y_test
            # Read in y_train
            filename = args.probMap_path + output_filenames[i] + '5'
            print('Test y set {0}/{1}'.format(i+1, test_nb, output_filenames[i]))
            # print('Training input: {0}'.format(filename))
            hf = h5py.File(filename, 'r')
            # Modified to switch order of indices
            data = np.transpose(hf.get('kmeans_clusters'), (2, 0, 1))  # [()] #.value
            y_test[i, :, :, :] = torch.tensor(data)
        
        # Channels
        inputs = torch.zeros((nb_tests, nb_channels, nx, ny))
        for j in range(nb_channels):
            inputs[0, j, :, :] = x_test[0, j, :, :]
        
        # Linnea: If freeze_encoder == 1, need to call net.forward2 with y_test instead of net.forward
        # Cuda
        if args.cuda:
            inputs = inputs.cuda()
            
            if args.freeze_encoder == 1:
                mask, outputs = net.forward2(inputs, y_test)
            else:
                mask, outputs = net.forward(inputs)
                
            inputs = inputs.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
        else:
            if args.freeze_encoder == 1:
                mask, outputs = net.forward2(inputs, y_test)
            else:
                mask, outputs = net.forward(inputs)

        # Outputs
        mask = mask.detach().cpu().numpy()
        if nb_channels == 1:
            inputs_tmp = np.zeros((nb_tests, nb_channels+1, nx, ny))
            inputs_tmp[0, 0, :, :] = inputs
            inputs_tmp[0, 1, :, :] = inputs
            new_mask = crf_batch_fit_predict(mask, inputs_tmp)
        else:
            new_mask = crf_batch_fit_predict(mask, inputs)
        label = mask.argmax(1)
        new_label = new_mask.argmax(1)

        # Save
        if args.test_save:
            filename = args.output_path+output_filenames[i]+'5'
            print(filename)
            with h5py.File(filename, 'w') as f:
                # f.create_dataset('origin', data=inputs)
                f.create_dataset('AE', data=outputs)
                f.create_dataset('Wnet', data=label)
                f.create_dataset('Wnet+CRF', data=new_label)
                f.close()

        # Visualization
        if args.visualise:

            print(r'---------------------- VISUALISING ----------------------')
            idx = np.random.randint(inputs.shape[0], size=(args.vis_num, ))
            visualise_outputs(inputs[idx], outputs[idx], label[idx], new_label[idx],
                              titles=['Origin', 'AE_recon', 'Wnet Mask',
                                      'Wnet+CRF Mask'])


