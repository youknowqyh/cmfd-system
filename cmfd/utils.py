from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
import numpy as np
import lmdb
import json
import cv2
# from core import create_cmfd_testing_model
import warnings
warnings.filterwarnings("ignore")

def gpu_test():
    '''
    Check if GPU is available
    '''
    import tensorflow as tf
    print("Num GPUS Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.test.gpu_device_name())

def bgr2rgb(bgr):
    # add back imageNet BGR means
    bgr = bgr + np.array([103.939, 116.779, 123.68]).reshape([1,1,3])
    rgb = bgr.astype(np.uint8)[:,:,::-1]
    return rgb

def image_preprocessing(rgb):
    """
    subtract ImageNet BGR pixel means
    """
    # Convert RGB into BGR
    bgr = rgb[:,:,::-1]
    # BGR mean values [103.94,116.78,123.68] are subtracted
    bgr = bgr - np.array([103.939, 116.779, 123.68]).reshape([1,1,3])
    return bgr

def cmfd_decoder(model, inp):
    '''
    inp -> input: Better to be BGR, but RGB is also feasible.
    '''
    inp = np.expand_dims(inp, axis=0)
    pred = model.predict(inp)[0]
    return pred

def visualize_result(rgb, gt, pred):
    """Visualize raw RGB input, ground truth, and pred cmfd mask
    """
    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes = axes.flatten()
    # add back imageNet BGR means
    axes[0].imshow(rgb)
    axes[0].set_title('Raw RGB input')
    axes[1].imshow(gt.astype(np.float32))
    axes[1].set_title('Ground truth')
    axes[2].imshow(pred)
    axes[2].set_title('Pred cmfd mask')

    for ax in axes:
        ax.axis('off')
    
    plt.show()

from sklearn.metrics import precision_recall_fscore_support
def pixel_level_evaluation():
    """
    - Only requires the positive samples
    - Computes (precision, recall, F1) scores for each positive sample, and then report the average scores over the entire dataset
    - Whatever the discernibility of source and target copies, so only the binary mask is used, i.e. the negate of the (blue) pristine channel.
    """
    pass

class CasiaUtils:
    def __init__(self, casia_dataset_path):
        """
        - these samples have already been processed (using keras's vgg16 preprocess_input)
        - all samples are resized to 256x256
        - since preprocess_input will convert RGB image to BGR, one shall reverse the color channel to display image properly
        """
        self.X = np.array(HDF5Matrix(casia_dataset_path, 'X'))
        self.Y = np.array(HDF5Matrix(casia_dataset_path, 'Y'))
    def __getitem__(self, key_idx):
        return self.get_one_sample(key=key_idx)
    def __len__(self):
        return len(self.X)
    def get_one_sample(self, key=None):
        return self.get_samples([key])[0]

    def get_samples(self, key_list):
        samples = []
        for key in key_list:
            if key is None:
                key = np.random.randint(0, len(self.X)-1)
                print("INFO: use random key", key)
  
            samples.append((self.X[key], self.Y[key]))
        
        return samples
    def visualize_by_index(self, idx):
        '''
        display single image and its ground truth
        '''
        inp = self.X[idx]
        out = self.Y[idx]
        fig, axes = plt.subplots(1, 4, figsize=(12,3))
        axes = axes.flatten()

        axes[0].imshow(inp.astype(np.uint8))
        axes[0].axis('off')
        axes[0].set_title('BGR preprocessed')
        
        axes[1].imshow(inp.astype(np.uint8)[:,:,::-1])
        axes[1].axis('off')
        axes[1].set_title('RGB preprocessed')

        # add back imageNet BGR means [103.939, 116.779, 123.68]
        inp = inp + np.array([103.939, 116.779, 123.68]).reshape([1,1,3])
        axes[2].imshow(inp.astype(np.uint8)[:,:,::-1])
        axes[2].axis('off')
        axes[2].set_title('Original RGB')

        axes[3].imshow(out.astype(np.float32)) # set ground truth to float
        axes[3].axis('off')
        axes[3].set_title('Ground truth')
        plt.show()


    def visualize_random_samples(self, batch_size=8):
        '''
        display a random batch of images

        self.X -> np.ndarray
        '''
        rows = batch_size // 4
        fig, axes = plt.subplots(rows, 4, figsize=(6, 1.5*rows))
        axes = axes.flatten()
        indices = np.random.choice(range(len(self.X)), batch_size)
        img_arr = self.X[indices]
        for idx, img, ax in zip(indices, img_arr, axes):
            ax.imshow(bgr2rgb(img))
            ax.axis('off')
            ax.set_title('index:' + str(idx))
        plt.tight_layout()
        plt.show()

class USCISIUtils:
    def __init__(self, lmdb_dir, sample_file, diff=True):
        self.lmdb_dir = lmdb_dir
        self.diff = diff
        with open(sample_file, 'r') as IN:
            self.sample_keys = [ line.strip() for line in IN.readlines() ]
        print("INFO: successfully load USC-ISI LMDB with {} keys".format(len(self.sample_keys)))

    def _get_mask_from_lut(self, lut):
        '''Decode copy-move mask from LMDB lut
        INPUT:
            lut = dict, raw decoded lut retrieved from LMDB
        OUTPUT:
            cmd_mask = np.ndarray, dtype='float32'
                       shape of HxWx1, if diff=False
                       shape of HxWx3, if diff=True
        NOTE:
            cmd_mask is encoded in the one-hot style, if diff=True.
            color channel, R, G, and B stand for TARGET, SOURCE, and BACKGROUND classes
        '''
        def reconstruct( cnts, h, w, val=1 ) :
            rst = np.zeros([h,w], dtype='uint8')
            cv2.fillPoly( rst, cnts, val )
            return rst 
        h, w = lut['image_height'], lut['image_width']
        cnts = lut['source_contour'][0]
        print(cnts)
        # src_cnts = [np.array(cnts).reshape([-1,1,2]) for cnts in lut['source_contour']]
    def visualize_samples(self, sample_list):
        fig, axes = plt.subplots(len(sample_list), 2, figsize=(6, 3*len(sample_list)))
        axes = axes.flatten()
        idx = 0
        for img, cmd_mask, trans_mat in sample_list:
            axes[idx].imshow(img)
            axes[idx+1].imshow(cmd_mask, cmap='gray')
            idx += 2
        plt.tight_layout()
        plt.show()

    def get_one_sample(self, key=None):
        return self.get_samples([key])[0]
    def get_samples(self, key_list):
        env = lmdb.open(self.lmdb_dir)
        sample_list = []
        with env.begin(write=False) as txn:
            for key in key_list:
                if key is None:
                    key = np.random.choice(self.sample_keys, 1)[0]
                    print("INFO: use random key", key)
                else:    
                    key = self.sample_keys[key]
                
                lut_str = txn.get(key.encode())
                # print(key)
                # 1. get raw lut
                lut = json.loads(lut_str)
                # print(lut.keys())
                # dict_keys(['transform_matrix', 'source_contour', 'image_width', 'target_contour', 'image_height', 'image_jpeg_buffer'])
                # 2. reconstruct image
                image_jpeg_buffer = lut['image_jpeg_buffer']
                image = cv2.imdecode(np.array(image_jpeg_buffer).astype('uint8'), cv2.IMREAD_COLOR)
                # If cv2.IMREAD_COLOR set, always convert image to the 3 channel BGR color image.
                # plt.imshow(image)
                # plt.show()
                # cv2.imshow('imshow',image)
                # cv2.waitKey(0)

                # get mask
                src_cnts = [np.array(lut['source_contour'][0]).reshape(-1,2)]
                tgt_cnts = [np.array(lut['target_contour'][0]).reshape(-1,2)]
                h, w = lut['image_height'], lut['image_width']
                src_mask = np.zeros([h,w], dtype='uint8')
                cv2.fillPoly(src_mask, src_cnts, 1)
                tgt_mask = np.zeros([h,w], dtype='uint8')
                cv2.fillPoly(tgt_mask, tgt_cnts, 1)
                # cv2.imshow('imshow', rst*255)
                # cv2.waitKey(0)
                # plt.imshow(rst, cmap='gray')
                # plt.show()
                if self.diff :
                    # 3-class target
                    background = np.ones([h,w]).astype('uint8') - np.maximum(src_mask, tgt_mask)
                    # this step is amazing..
                    # tgt_mask, src_mask and background respectively occupies one channel..
                    # and dstack will concatenate along Z axis...
                    cmd_mask = np.dstack( [tgt_mask, src_mask, background ] ).astype(np.float32)
                else :
                    # 2-class target
                    cmd_mask = np.maximum(src_mask, tgt_mask).astype(np.float32)
                # 
                trans_mat = np.array(lut['transform_matrix']).reshape([3,3])
                sample = (image, cmd_mask, trans_mat)
                sample_list.append(sample)
        return sample_list
    def __call__(self, key_list):
        return self.get_samples(key_list)
    def __len__(self):
        return len(self.sample_keys)
    def __getitem__(self, key_idx):
        return self.get_one_sample(key=key_idx)
        
def casia_test():
    cmfd_model = create_cmfd_testing_model('pretrained.hd5')
    casia = CasiaUtils('CASIA-CMFD-Pos.hd5')
    # Z = cmfd_model.predict(casia.X, verbose=1)

    # example 1
    # casia.visualize_by_index(idx=5)

    # example 2: let's show 16 random images from the casia tide database.
    # casia.visualize_random_samples(batch_size=16)
    
    
    # example 3: Visualize raw input, ground truth, and pred cmfd mask
    sample_idx = 20
    rgb = bgr2rgb(casia.X[sample_idx])
    gt = casia.Y[sample_idx] # ground truth
    pred = cmfd_decoder(cmfd_model, casia.X[sample_idx])
    visualize_result(rgb, gt, pred)

def uscisi_test():
    lmdb_dir = r'H:\数据集\USCISI-CMFD'
    sample_file = r'H:\数据集\USCISI-CMFD\samples.keys'

    # create dataset instance
    dataset = USCISIUtils(lmdb_dir, sample_file)

    # retrieve the first 12 samples in the dataset
    samples = dataset(range(3))
    print(len(samples))

    # retrieve 12 random samples in the dataset
    # random_samples = dataset( [None]*24 )
    
    # visualize these samples
    dataset.visualize_samples(samples)
if __name__ == '__main__':
    # casia_test()
    uscisi_test()