from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset, Movies_Dataset, Movies_Dataset_v2
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

    #  173  (8000)    
    #  35   (fh)  
    #  86   (Christmas)   
    #  4    (3 films--[Y,C,BP])    
    #  2    (5 films--[Y,C,BP,YW,Q])   
    #  48   (3 films--[BP,YW,Q])
    #  4    (5 films--[Y,C,BP,YW,Q])
    
    #  173  (8000)         
    #  35   (fh)       
    #  86   (Christmas) 
    #  24   (5 fimes [BP,YW,Q,007,AQW])
    
    
def main():
    # Configue
    resume_path = './lightning_logs/version_34/checkpoints/epoch=85-step=117389.ckpt'
    data_path = '/home/dailongquan/110.015/datasets/moves_data/movies'
    txt_path = '/home/dailongquan/110.015/datasets/moves_data/base'
    model_path = './models/cldm_v21_color_v2.yaml'
    batch_size = 15
    logger_freq = 5000
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_path).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    # dataset = MyDataset()
    dataset = Movies_Dataset_v2(data_path, txt_path, 0.97, 3)
    dataloader = DataLoader(dataset, num_workers=6, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=[1,2,3,4], precision=32, accelerator='ddp',callbacks=[logger])

    # Train!
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    main()

