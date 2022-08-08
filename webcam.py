import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
import numpy as np
import cv2


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    model = create_model(opt)
    model.setup(opt)
    if opt.eval:
        model.eval()

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise IOError("Cannot open webcam")

    style_models = ['style_monet_pretrained', 'style_vangogh_pretrained', 'style_ukiyoe_pretrained', 'style_cezanne_pretrained']
    style_model_index = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, 25)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    data = {"A": None, "A_paths": None}
    while True: 
        ret, frame = webcam.read()

        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = np.array([frame])
        frame = frame.transpose([0,3,1,2])

        data['A'] = torch.FloatTensor(frame)
        
        model.set_input(data)
        model.test()

        #get only generated image - indexing dictionary for "fake" key
        result_image = model.get_current_visuals()['fake']
        #use tensor2im provided by util file
        result_image = util.tensor2im(result_image)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_BGR2RGB)  
        result_image = cv2.resize(result_image, (512, 512))      
        result_image = cv2.putText(result_image, str(opt.name)[6:-11], org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)   
        cv2.imshow('style', result_image)

        #ASCII value of Esc is 27.
        c = cv2.waitKey(1)
        if c == 27:
            break
        if c == 99:
            if style_model_index == len(style_models):
                style_model_index = 0
            opt.name = style_models[style_model_index]
            style_model_index += 1
            model = create_model(opt)      # create a model given opt.model and other options
            model.setup(opt) 
      
        
    cap.release()
    cv2.destroyAllWindows()
        