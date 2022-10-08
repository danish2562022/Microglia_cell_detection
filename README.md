
# Microglia quantification 

In this project, we automate quantification of microglia.



## Getting Started


### Installing


Say what the step will be
    
    #### Run in the Terminal(search in the mac/linux/windows) 

    $ git clone https://github.com/danish2562022/Microglia_cell_detection.git
    $ cd Microglia_cell_detection
    $ pip install -r requirements.txt

## Configuration file
Directories and Models are defined in config.yaml
    
    experiment_name: Microglia_object_detection
    Model_information:
        Model_name: "FASTER-RCNN" 
        Threshold: 0.5
        NMS: 0.1
        image_format: "tif"     
    Directory:
        Input_dir: "./test_images"
        Output_dir: "./results"
    Weights:
        faster-rnn_weights: "./Faster_RCNN_weights/model_final.pth"
        retinanet_weights: "./RetinaNet_weights/model_final.pth"

-->Model_name: Name of the model<br />
            -- Write "FASTER-RCNN" for Faster-RCNN<br />
            -- Write "RETINA-NET" for RetinaNet<br /><br />
-->Directory: Input_dir is the path of input images<br />
              Output_dir is the path of results(Detected images and excel sheet of quantification) 
            
## Running

How to run the automated hyperparameter optimization pipeline(https://www.youtube.com/watch?v=kCfhfJxXOOA&ab_channel=Danish)<br />

    $ cd Microglia_cell_detection
    $ python inference.py








        

## Built With

 

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.



## License



## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
