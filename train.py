import tensorflow as tf
import numpy as np

from transformers import AutoTokenizer
import time
import argparse

from e2e_transformers.model import E2ETransformer
from e2e_transformers.lr_scheduler import CustomSchedule
from data_preprocessing import preprocessing_py_func
from e2e_transformers.utils import create_masks
from e2e_transformers.utils import loss_function


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, slot_inp, tar):

    # inp = x[0]
    # slot_inp = x[1]
    # tar = x[2]
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    
    with tf.GradientTape() as tape:
        predictions, _ = e2e_model(inp, slot_inp, tar_inp, 
                                    True, 
                                    enc_padding_mask, 
                                    combined_mask, 
                                    dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, e2e_model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, e2e_model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def train(train_data, optimizer, opt, ckpt_manager):

    global train_loss, train_accuracy
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
    epoch = opt.epoch

    for epoch in range(epoch):
        start = time.time()
        
        train_loss.reset_states()
        train_accuracy.reset_states()
  
        # inp -> portuguese, tar -> english
        for (batch, x) in enumerate(train_data):
            train_step(*x)

            if batch % 50 == 0:
                print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_manager.save()
            print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                            ckpt_save_path))
    
        print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                    train_loss.result(), 
                                                    train_accuracy.result()))

        print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def prepare_data(opt):
    train_data = tf.data.experimental.CsvDataset(filenames=opt.train_path, 
                                    record_defaults=[tf.string, tf.string],
                                    header=True,
                                    select_cols=[0, 1])

    train_data = train_data.map(preprocessing_py_func)\
        .shuffle(buffer_size=opt.buffer)\
        .batch(opt.batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_data


def main():
    ''' 
    Usage:
    python train.py -data_pkl m30k_deen_shr.pkl -log m30k_deen_shr -embs_share_weight -proj_share_weight -label_smoothing -save_model trained -b 256 -warmup 128000
    '''

    parser = argparse.ArgumentParser()


    parser.add_argument('-train_path', default=None)    
    parser.add_argument('-val_path', default='val_data.csv')    

    parser.add_argument('-epoch', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)

    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=1024)

    parser.add_argument('-embedding', type=str, default='t5_extended_embed.npy')

    parser.add_argument('-n_heads', type=int, default=4)
    parser.add_argument('-n_enc_layers', type=int, default=3)
    parser.add_argument('-n_dec_layers', type=int, default=5)
    parser.add_argument('-max_len', type=int, default=100)

    parser.add_argument('-warmup','--n_warmup_steps', type=int, default=4000)
    parser.add_argument('-pad_idx', type=int, default=0)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-buffer', type=int, default=20000)
    parser.add_argument("-new_opt", type=bool, action="store_true")


    opt = parser.parse_args()

    global e2e_model
    e2e_model = E2ETransformer.from_config(opt)

    embedding_weight = np.load(opt.embedding)
    vocab_size, d_model = embedding_weight.shape

    learning_rate = CustomSchedule(d_model)

    global optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
    
    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(e2e_model = e2e_model,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')


    #========= Loading Dataset =========#
    train_data = prepare_data(opt)
    if opt.new_opt:
        learning_rate = CustomSchedule(d_model)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)


    train(train_data, optimizer, opt, ckpt_manager)




if __name__ == '__main__':
    main()
